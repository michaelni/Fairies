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

Regression test for ``run_cmd``'s ``stderr_line_prefix`` mode.

Earlier versions of ``run_cmd`` combined a custom stderr pump thread
(``for line in proc.stderr``) with ``proc.communicate(input=...)``.
On POSIX, ``communicate()`` spawns its own internal stderr-draining
thread that calls ``os.read(stderr_fd, ...)`` on the same pipe the
custom pump is reading. ``os.read`` is atomic per chunk, so each
chunk randomly went to whichever thread won the race. The losers'
chunks ended up in ``communicate``'s discarded ``stderr`` return
value and silently vanished from the operator's log file.

This test reproduces that race by spawning a subprocess that writes
many short lines to stderr in a tight loop, and asserts that every
single line emitted by the child is forwarded by ``run_cmd``'s pump
-- through the logger, so the lines also reach the ``--log-file``
handlers the fairy-ui pane tails, not just the process console.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import forge_gcli  # noqa: E402


_CHILD_SCRIPT = (
    "import sys\n"
    "n = int(sys.argv[1])\n"
    # Echo stdin to stdout so we can also check stdout still works.
    "data = sys.stdin.read()\n"
    "sys.stdout.write(data)\n"
    "sys.stdout.flush()\n"
    "for i in range(n):\n"
    "    sys.stderr.write(f'line-{i:05d}\\n')\n"
    "sys.stderr.flush()\n"
)


class RunCmdStderrPumpTest(unittest.TestCase):
    def test_no_lines_dropped_and_all_reach_the_logger(self) -> None:
        n_lines = 2000
        with self.assertLogs(forge_gcli.logger, level="INFO") as logs:
            cp = fairy.run_cmd(
                [sys.executable, "-c", _CHILD_SCRIPT, str(n_lines)],
                input_text="hello-stdin\n",
                stderr_line_prefix="[child] ",
            )
        self.assertEqual(cp.returncode, 0)
        # stdout round-tripped through the child unchanged.
        self.assertEqual(cp.stdout, "hello-stdin\n")
        # Every stderr line emitted by the child was prefixed and
        # forwarded by the pump (no chunks lost to a racing reader).
        observed = [r.getMessage() for r in logs.records
                    if r.getMessage().startswith("[child] ")]
        expected = [f"[child] line-{i:05d}" for i in range(n_lines)]
        self.assertEqual(observed, expected)


class GcliTimeoutTests(unittest.TestCase):
    def test_every_gcli_call_is_bounded(self) -> None:
        with mock.patch.object(forge_gcli, "run_cmd") as run_cmd, \
                mock.patch.object(forge_gcli.github_app, "gcli_env",
                                  return_value=None):
            forge_gcli.run_gcli(mock.Mock(verbose=0), ["gcli", "api", "/user"])
        self.assertEqual(forge_gcli.GCLI_TIMEOUT_S,
                         run_cmd.call_args.kwargs["timeout"])


class WireRelayTests(unittest.TestCase):
    """A child logging in FAIRY_LOG_WIRE shape keeps level and time
    through the relay: the pane colors wrapper errors red, and exactly
    one timestamp (the child's own) survives per line."""

    def test_wire_lines_keep_their_level_and_time(self) -> None:
        script = (
            "import sys\n"
            "sys.stderr.write('2026-07-24T10:00:01 E it broke\\n')\n"
            "sys.stderr.write('plain transcript line\\n')\n")
        with self.assertLogs(forge_gcli.logger, level="DEBUG") as logs:
            fairy.run_cmd([sys.executable, "-c", script],
                          stderr_line_prefix="[w] ")
        wire = next(r for r in logs.records if "it broke" in r.getMessage())
        self.assertEqual(wire.levelno, logging.ERROR)
        self.assertEqual(wire.getMessage(), "[w] it broke")
        self.assertEqual(
            datetime.fromtimestamp(wire.created).strftime("%Y-%m-%dT%H:%M:%S"),
            "2026-07-24T10:00:01")
        plain = next(r for r in logs.records
                     if "transcript" in r.getMessage())
        self.assertEqual(plain.levelno, logging.INFO)

    def test_the_wrapper_child_is_asked_for_the_wire_format(self) -> None:
        ns = fairy.parse_args(["--owner", "o", "--repo", "r",
                               "--llm-review-cmd", "wrapper",
                               "--patch-repo", "p"])
        done = subprocess.CompletedProcess(
            [], 0, stdout='{"classification": "skip", "message": ""}',
            stderr="")
        with mock.patch.object(fairy, "run_cmd", return_value=done) as rc:
            fairy.invoke_llm_wrapper(
                ns, {}, number=1,
                allowed_classifications=frozenset({"skip"}),
                label_allowlist=[])
        self.assertEqual(rc.call_args.kwargs["env"]["FAIRY_LOG_WIRE"], "1")


if __name__ == "__main__":
    unittest.main()
