"""CodexContainer: ``podman exec`` argv construction over RemoteHost ssh."""

import io
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import codex_container
from codex_container import CodexContainer, CodexShellRelay

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


class _FakeProc:
    def __init__(self, stderr_bytes: bytes):
        self.stderr = io.BytesIO(stderr_bytes)
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


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


if __name__ == "__main__":
    unittest.main()
