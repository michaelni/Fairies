#!/usr/bin/env python3
"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License version 2 for more details.
 *
 * Additional permission:
 *
 * Michael Niedermayer is permitted to relicense this file, in whole or
 * in part, under any version of the GNU General Public License, the GNU
 * Affero General Public License, or the GNU Lesser General Public License
 * published by the Free Software Foundation.
 *
 * This additional permission is personal to Michael Niedermayer.  It is
 * not transferable and does not grant any other person permission to
 * relicense this file under a different license.
 *
 * This additional permission may be removed from modified copies of this
 * file.  Removal of this additional permission does not affect the
 * licensing of the file under the GNU General Public License version 2.
 */

In-container shell agent for fairy reviews.

Runs inside the ephemeral review container. Reads one JSON request per
line from stdin, runs it under ``bash -lc`` with a watchdog timeout, and
writes one JSON response per line to stdout. The local runner holds the
other end of a single persistent ``ssh ... podman exec -i`` pipe, so
there is no per-command ssh handshake and commands cross as data (no
shell re-tokenisation).

stdout carries ONLY protocol frames (flushed per reply); every
diagnostic goes to stderr so the stream the runner parses can never be
corrupted by a stray print.

Protocol (newline-delimited JSON; all binary travels base64, so a frame
never contains a raw newline):

  request:  {"id", "command", "cwd"|null, "timeout_s", "max_output_bytes"?}
  response: {"id", "exit_code", "stdout_b64", "stderr_b64",
             "stdout_truncated", "stderr_truncated", "duration_s"}

``exit_code`` follows the ``timeout(1)`` convention: 124 means the
watchdog fired and SIGKILLed the command's process group; 127 means the
command could not be launched.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time

DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024
TIMEOUT_EXIT_CODE = 124
SPAWN_FAILED_EXIT_CODE = 127
_KILL_GRACE_S = 5.0
_PUMP_JOIN_GRACE_S = 5.0
_PUMP_DRAIN_GRACE_S = 0.5


def _pump_capped(stream, out_buf: bytearray, cap: int, truncated: list[bool]) -> None:
    """Read ``stream`` to EOF, keeping at most ``cap`` bytes; discard the
    rest so the child never blocks on a full pipe."""
    while True:
        chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
        if not chunk:
            return
        room = cap - len(out_buf)
        if room > 0:
            out_buf.extend(chunk[:room])
        if len(chunk) > room:
            truncated[0] = True


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then (after a grace period) SIGKILL the child's whole
    process group, so a timed-out command leaves no orphans behind."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=_KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def run_command(command: str, cwd: str | None, timeout_s: float,
                max_output_bytes: int) -> dict:
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=cwd or None,
            # DEVNULL, never inherited: the agent's own stdin is the
            # protocol pipe, and an stdin-reading command (ffmpeg without
            # -nostdin, cat, ...) would eat protocol frames from it.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return _result(SPAWN_FAILED_EXIT_CODE, b"", str(exc).encode(),
                       False, False, time.monotonic() - t0)

    out_buf, err_buf = bytearray(), bytearray()
    out_trunc, err_trunc = [False], [False]
    threads = [
        threading.Thread(target=_pump_capped,
                         args=(proc.stdout, out_buf, max_output_bytes, out_trunc),
                         daemon=True),
        threading.Thread(target=_pump_capped,
                         args=(proc.stderr, err_buf, max_output_bytes, err_trunc),
                         daemon=True),
    ]
    for t in threads:
        t.start()

    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        exit_code = TIMEOUT_EXIT_CODE
        timed_out = True
    # A process that moved to its own group (GNU timeout(1), setsid, a
    # backgrounded daemon) survives the group kill and keeps the output
    # pipes open, so waiting for pipe EOF could delay the reply past the
    # runner's response deadline and get the whole channel declared dead.
    # Give the pumps a bounded grace and answer with what was captured:
    # after a clean exit the holder is a deliberate background job, so
    # drain briefly and report the output complete; after a kill it may
    # still have been producing, hence truncated.
    deadline = time.monotonic() + (
        _PUMP_JOIN_GRACE_S if timed_out else _PUMP_DRAIN_GRACE_S)
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))

    # An abandoned pump keeps its pipe fd and its thread until the
    # surviving writer exits -- a daemon left running leaks both for the
    # agent's lifetime. Needs fixing: a raw-fd reader whose close is
    # safe under a concurrent read.
    return _result(exit_code, bytes(out_buf), bytes(err_buf),
                   out_trunc[0] or (timed_out and threads[0].is_alive()),
                   err_trunc[0] or (timed_out and threads[1].is_alive()),
                   time.monotonic() - t0)


def _result(exit_code: int, stdout: bytes, stderr: bytes,
            stdout_truncated: bool, stderr_truncated: bool,
            duration_s: float) -> dict:
    return {
        "exit_code": exit_code,
        "stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "duration_s": duration_s,
    }


def _handle_request(line: bytes) -> dict:
    """Parse one request line and run it. Always returns a well-formed
    response (a bad request maps to exit_code 127) so the runner's reader
    never desyncs."""
    req = json.loads(line)
    command = req["command"]
    return {
        "id": req.get("id"),
        **run_command(
            command,
            req.get("cwd"),
            float(req.get("timeout_s", 120.0)),
            int(req.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES)),
        ),
    }


def main() -> int:
    out = sys.stdout.buffer
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        if not line.strip():
            continue
        try:
            response = _handle_request(line)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"fairy_agent: bad request: {exc!r}", file=sys.stderr, flush=True)
            response = {"id": None,
                        **_result(SPAWN_FAILED_EXIT_CODE, b"", str(exc).encode(),
                                  False, False, 0.0)}
        out.write((json.dumps(response) + "\n").encode("utf-8"))
        out.flush()


if __name__ == "__main__":
    raise SystemExit(main())
