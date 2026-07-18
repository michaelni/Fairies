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

Shell dispatch across the codex process boundary, over a per-run unix
socket (local IPC only -- no TCP, nothing network-facing).

The codex CLI spawns its MCP servers itself, so the MCP bridge
(``codex_bridge.py``) runs outside the wrapper process that owns the live
container sessions and their lifecycle. This module carries one shell
tool call across that boundary: the wrapper side listens on a unix socket
inside a mode-0700 temp directory and dispatches each request through
``shell_tool.exec_machine_call``; the bridge side holds one connection
for the lifetime of its codex run.

One connection = one codex run = one private ``shells`` dict, so each
codex pass gets its own lazily-opened per-machine containers -- the same
isolation ``run_parallel`` gives the other backends. Sessions opened here
are registered for cleanup by the wrapper-supplied ``open_shell``
callback, exactly like every other reviewer's.

Wire protocol: newline-delimited JSON, one request/response pair at a
time per connection. Request ``{"id": n, "args": {...}}`` where ``args``
is the model's shell-tool input; response ``{"id": n, "payload": {...}}``
where ``payload`` is exec_machine_call's result (or an in-band
``{"error": ...}`` the model can read, like every other shell backend).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
from typing import Callable, Sequence

from common import JsonObject
from podman_host import ContainerShellSession
from shell_tool import exec_machine_call
# ShellDispatchClient (the codex-container side) lives in the podman-free
# shell_bridge_client so the codex image can import it without this module;
# re-exported here for the wrapper-side callers and tests.
from shell_bridge_client import ShellDispatchClient, _recv_json_line

__all__ = ["ShellDispatchClient", "ShellDispatchServer", "serve_dispatch"]

logger = logging.getLogger(__name__)

SOCKET_NAME = "shell.sock"


def _write_json(writer, obj: JsonObject) -> None:
    writer.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
    writer.flush()


def serve_dispatch(
    reader,
    writer,
    *,
    machine_labels: Sequence[str],
    open_shell: Callable[[str], tuple[ContainerShellSession, str]],
    max_timeout_s: float,
) -> None:
    """Serve one codex run's shell dispatch over a byte-stream pair.

    ``reader``/``writer`` are binary file-like objects carrying the same
    newline-delimited ``{"id", "args"}`` / ``{"id", "payload"}`` protocol
    as the unix socket -- but transport-agnostic, so the wrapper can drive
    it over a ``podman exec -i`` channel into the codex container (the
    ``relay.py`` side) exactly as it drives the review shells, no socket of
    its own required.

    One call == one codex run: a private ``shells`` dict, so a run never
    shares a container working tree with a concurrent pass. Returns on EOF
    (``reader`` exhausted).
    """
    shells: dict[str, ContainerShellSession] = {}
    while True:
        try:
            request = _recv_json_line(reader)
        except json.JSONDecodeError as exc:
            logger.warning("shell dispatch: bad request line: %s", exc)
            _write_json(writer, {"id": None, "payload": {
                "error": f"malformed request: {exc}"}})
            continue
        if request is None:
            return
        try:
            payload = exec_machine_call(
                shells, tuple(machine_labels), open_shell,
                request.get("args"), max_timeout_s=max_timeout_s,
            )
        except Exception as exc:
            logger.exception("shell dispatch failed")
            payload = {"error": f"shell dispatch failed: {exc}"}
        _write_json(writer, {"id": request.get("id"), "payload": payload})


class ShellDispatchServer:
    """Wrapper-side unix-socket server dispatching shell tool calls.

    ``open_shell`` is the wrapper's ``open_machine_shell`` (registers each
    opened container for the wrapper's own cleanup); ``machine_labels[0]``
    is the default machine, as everywhere. ``close()`` stops accepting and
    removes the socket; already-running commands finish on their
    connection threads, whose sessions the wrapper's finally releases.
    """

    def __init__(
        self,
        *,
        machine_labels: Sequence[str],
        open_shell: Callable[[str], tuple[ContainerShellSession, str]],
        max_timeout_s: float,
    ) -> None:
        self.machine_labels = tuple(machine_labels)
        self.open_shell = open_shell
        self.max_timeout_s = max_timeout_s
        # The socket path is the access token: the directory mode is the
        # only thing keeping other local users from the shell dispatch.
        self._dir = tempfile.mkdtemp(prefix="fairy-shell-")
        os.chmod(self._dir, 0o700)
        self.socket_path = os.path.join(self._dir, SOCKET_NAME)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.socket_path)
        self._listener.listen()
        self._closed = False
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="shell-dispatch-accept", daemon=True,
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:  # listener closed
                return
            threading.Thread(
                target=self._serve_connection, args=(conn,),
                name="shell-dispatch-conn", daemon=True,
            ).start()

    def _serve_connection(self, conn: socket.socket) -> None:
        # Same dispatch loop as the codex-container relay, over this
        # connection's byte streams (see serve_dispatch).
        reader = conn.makefile("rb")
        writer = conn.makefile("wb")
        try:
            serve_dispatch(
                reader, writer,
                machine_labels=self.machine_labels,
                open_shell=self.open_shell,
                max_timeout_s=self.max_timeout_s,
            )
        except (BrokenPipeError, ConnectionResetError):
            logger.info("shell dispatch: connection dropped")
        finally:
            reader.close()
            writer.close()
            conn.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._listener.close()
        finally:
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
            try:
                os.rmdir(self._dir)
            except OSError:
                pass
