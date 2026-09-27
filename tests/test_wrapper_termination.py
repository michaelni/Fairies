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

pr_review_wrapper.stop_containers_on_termination: a wrapper process removes
its containers on SIGTERM and when its parent process exits."""

import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WRAPPER_STANDIN = f"""
import sys, time
from pathlib import Path
from unittest import mock
sys.path.insert(0, {str(REPO_ROOT)!r})
import podman_host, pr_review_wrapper

def fake_podman(host, *args, timeout_s):
    with open(sys.argv[1], "a") as podman_log:
        podman_log.write(" ".join(args) + "\\n")
    return mock.Mock(returncode=0, stdout=b"cid1\\n", stderr=b"")

podman_host._podman = fake_podman
podman_host.default_cache_path = Path(sys.argv[1]).parent.joinpath
podman_host.start_ephemeral_container(image="img", host=podman_host.RemoteHost("fairy@h"))
pr_review_wrapper.stop_containers_on_termination()
print("ready", flush=True)
time.sleep(60)
"""

WORKER_STANDIN = """
import subprocess, sys
wrapper = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]], stdout=subprocess.PIPE, text=True)
print(wrapper.pid, wrapper.stdout.readline().strip(), flush=True)
"""


@unittest.skipIf(sys.platform == "win32", "SIGTERM and parent-exit detection are POSIX")
class StopContainersOnTerminationTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.podman_log = Path(tmp.name) / "podman.log"

    def _removed_containers(self) -> list[str]:
        return [line for line in self.podman_log.read_text().splitlines()
                if line.startswith("rm -f")]

    def test_sigterm_removes_containers_and_exits_143(self) -> None:
        wrapper = subprocess.Popen(
            [sys.executable, "-c", WRAPPER_STANDIN, str(self.podman_log)],
            stdout=subprocess.PIPE, text=True)
        self.assertEqual("ready", wrapper.stdout.readline().strip())
        wrapper.send_signal(signal.SIGTERM)
        self.assertEqual(143, wrapper.wait(30))
        wrapper.stdout.close()
        self.assertEqual(["rm -f cid1"], self._removed_containers())

    def test_parent_exit_removes_containers_and_exits(self) -> None:
        worker = subprocess.run(
            [sys.executable, "-c", WORKER_STANDIN, WRAPPER_STANDIN, str(self.podman_log)],
            stdout=subprocess.PIPE, text=True, timeout=60, check=True)
        wrapper_pid, state = worker.stdout.split()
        self.addCleanup(lambda: subprocess.run(["kill", "-9", wrapper_pid], capture_output=True))
        self.assertEqual("ready", state)
        deadline = time.monotonic() + 30
        while not self._removed_containers() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertEqual(["rm -f cid1"], self._removed_containers())

if __name__ == "__main__":
    unittest.main()
