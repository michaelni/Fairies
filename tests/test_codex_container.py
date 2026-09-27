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

CodexContainer: ``podman exec`` argv construction over RemoteHost ssh."""

import io
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import codex_container
from codex_container import CodexContainer, CodexShellRelay
from test_podman_host import ContainerStateHarness
from common import EXIT_REVIEW_STOPPED_BY_HALT_FILE

HOST = podman_host.RemoteHost("fairy@box", identity="/id")


class ExecArgvTests(unittest.TestCase):
    def _container(self):
        c = CodexContainer(image="localhost/fairy-codex:latest", host=HOST)
        c.handle = podman_host.ContainerHandle(
            container_id="abc123", image="img", network=None, host=HOST)
        return c

    def test_exec_argv_wraps_podman_exec_over_ssh(self) -> None:
        argv = self._container().exec_argv(["codex", "--version"])
        self.assertEqual("ssh", argv[0])
        self.assertIn("fairy@box", argv)
        self.assertIn("-i", argv)  # ssh identity flag
        self.assertIn("/id", argv)
        # RemoteHost shlex-joins the remote command into the final arg.
        self.assertEqual("podman exec abc123 codex --version", argv[-1])

    def test_interactive_and_env_flags(self) -> None:
        argv = self._container().exec_argv(
            ["codex", "exec", "-"], interactive=True,
            env={"CODEX_HOME": codex_container.CONTAINER_CODEX_HOME},
        )
        self.assertEqual(
            "podman exec -i -e CODEX_HOME=/work/.codex-home abc123 "
            "codex exec -",
            argv[-1],
        )

    def test_read_file_cats_container_path(self) -> None:
        argv = self._container().exec_argv(["cat", "/work/.codex-run/out.json"])
        self.assertEqual(
            "podman exec abc123 cat /work/.codex-run/out.json", argv[-1])

    def test_read_file_max_bytes_uses_head(self) -> None:
        c = self._container()
        with mock.patch.object(codex_container.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="{}")
            c.read_file("/work/.codex-run/last_message.json", max_bytes=1024)
        self.assertEqual(
            "podman exec abc123 head -c 1024 /work/.codex-run/last_message.json",
            run.call_args.args[0][-1])

    def test_exec_before_start_is_error(self) -> None:
        c = CodexContainer(image="img", host=HOST)
        with self.assertRaises(RuntimeError):
            c.exec_argv(["codex"])


class KillRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.container = CodexContainer(image="img", host=HOST)
        self.container.handle = podman_host.ContainerHandle(
            container_id="cid", image="img", network=None, host=HOST)

    def _kill_during_run(self, **kill_effect) -> tuple[mock.Mock, mock.Mock]:
        with mock.patch.object(codex_container, "run_on_remote_host",
                               **kill_effect) as kill, \
                mock.patch.object(codex_container, "stop_container") as stop, \
                mock.patch.object(codex_container.subprocess, "Popen",
                                  return_value=_FakeProc(
                                      b"", on_communicate=self.container.kill_run)):
            self.container.run(["codex"])
        return kill, stop

    def test_kill_during_a_run_spares_pid_1_and_the_container(self) -> None:
        for rc in (0, 1):
            with self.subTest(rc=rc):
                kill, stop = self._kill_during_run(
                    return_value=mock.Mock(returncode=rc))
                kill.assert_called_once_with(
                    HOST, "podman", "exec", "cid", "sh", "-c", "kill -KILL -1",
                    timeout_s=60.0)
                stop.assert_not_called()

    def test_failed_kill_removes_the_container(self) -> None:
        for effect in (
                {"return_value": mock.Mock(returncode=125)},
                {"side_effect": codex_container.subprocess.TimeoutExpired(
                    "ssh", 60.0)}):
            with self.subTest(effect=effect):
                self.setUp()
                _, stop = self._kill_during_run(**effect)
                stop.assert_called_once()

    def test_no_kill_outside_a_run(self) -> None:
        with mock.patch.object(codex_container, "run_on_remote_host") as kill:
            self.container.kill_run()
        kill.assert_not_called()

    def test_stop_removes_the_container_once(self) -> None:
        with mock.patch.object(codex_container, "stop_container") as stop:
            self.container.stop()
            self.container.stop()
        stop.assert_called_once()


class RunTests(ContainerStateHarness, unittest.TestCase):
    def test_pausing_the_container_ends_a_codex_that_never_returns(self) -> None:
        self.record_podman()
        c = CodexContainer(image="img", host=HOST)
        codex = [sys.executable, "-c", "import time; time.sleep(60)"]
        with mock.patch.object(c, "exec_argv", return_value=codex):
            c.start()
            pause = threading.Timer(0.5, podman_host.pause_container, [c.handle])
            pause.start()
            started = time.monotonic()
            proc = c.run(["codex", "exec"], input_text="prompt")
            pause.join()
        self.assertLess(time.monotonic() - started, 30)
        self.assertNotEqual(0, proc.returncode)


class _FakeProc:
    def __init__(self, stderr_bytes: bytes, on_communicate=lambda: None):
        self.stderr = io.BytesIO(stderr_bytes)
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.killed = False
        self.args = []
        self.returncode = None
        self._on_communicate = on_communicate

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def communicate(self, input=None, timeout=None):
        self._on_communicate()
        return "", ""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class RelayReadinessTests(unittest.TestCase):
    """CodexShellRelay.start scans stderr for RELAY-READY past ssh/podman
    warning lines instead of trusting the first line."""

    def _relay(self):
        c = CodexContainer(image="img", host=HOST)
        c.handle = podman_host.ContainerHandle(
            container_id="abc", image="img", network=None, host=HOST)
        return CodexShellRelay(
            c, relay_container_path="/work/.codex-run/relay.py",
            machine_labels=("x86_64",), open_shell=lambda label: (None, ""),
            max_timeout_s=60.0)

    def _start_with_stderr(self, stderr_bytes):
        relay = self._relay()
        proc = _FakeProc(stderr_bytes)
        with mock.patch.object(codex_container.subprocess, "Popen",
                               return_value=proc), \
                mock.patch.object(codex_container, "serve_dispatch"):
            relay.start(ready_timeout_s=5.0)
        return relay, proc

    def test_ready_after_warning_line(self) -> None:
        # ssh prints a host-key warning before relay.py's marker.
        relay, proc = self._start_with_stderr(
            b"Warning: Permanently added 'box' to known hosts.\n"
            b"RELAY-READY\n")
        self.assertIsNotNone(relay._thread)  # dispatch started -> ready seen

    def test_ready_scan_buffer_is_capped(self) -> None:
        noise = (b"x" * 20000 + b"\n") * 10
        relay, proc = self._start_with_stderr(noise + b"RELAY-READY\n")
        self.assertIsNotNone(relay._thread)  # marker still seen
        self.assertLessEqual(sum(map(len, relay._ready_lines)), 2 * 65536)

    def test_dead_relay_raises_not_hangs(self) -> None:
        relay = self._relay()
        proc = _FakeProc(b"some podman error\n")  # EOF, no RELAY-READY
        with mock.patch.object(codex_container.subprocess, "Popen",
                               return_value=proc), \
                mock.patch.object(codex_container, "serve_dispatch"):
            with self.assertRaisesRegex(RuntimeError, "relay failed to start"):
                relay.start(ready_timeout_s=5.0)
        self.assertTrue(proc.killed)


class RelayCancelTests(unittest.TestCase):
    def _serve(self, dispatch_effect) -> mock.Mock:
        container = mock.Mock()
        relay = CodexShellRelay(
            container, relay_container_path="/work/.codex-run/relay.py",
            machine_labels=("x86_64",), open_shell=lambda label: (None, ""),
            max_timeout_s=60.0)
        relay._proc = _FakeProc(b"")
        with mock.patch.object(codex_container, "serve_dispatch",
                               side_effect=dispatch_effect):
            relay._serve()
        return container

    def test_cancel_at_a_shell_call_kills_codex_and_keeps_the_container(self) -> None:
        container = self._serve(SystemExit)
        container.kill_run.assert_called_once()
        container.stop.assert_not_called()

    def test_a_halt_leaves_the_codex_container_to_the_halt_pause(self) -> None:
        container = self._serve(SystemExit(EXIT_REVIEW_STOPPED_BY_HALT_FILE))
        container.kill_run.assert_not_called()
        container.stop.assert_not_called()

    def test_normal_eof_stops_nothing(self) -> None:
        self._serve(None).kill_run.assert_not_called()

    def test_closed_relay_pipe_neither_raises_nor_stops(self) -> None:
        """A shell call that outlives codex writes its response to the
        stopped relay's closed stdin; the dispatch thread must end quietly
        instead of dying with the traceback seen in production 2026-08-14."""
        self._serve(ValueError("write to closed file")).kill_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
