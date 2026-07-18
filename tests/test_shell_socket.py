"""Shell dispatch over the per-run unix socket (codex backend plumbing)."""

import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import shell_socket

RELAY_PY = Path(__file__).resolve().parent.parent / "containers" / "relay.py"


def _result(out=""):
    return podman_host.ExecResult(
        exit_code=0, stdout=out, stderr="", duration_s=0.01,
        stdout_truncated=False, stderr_truncated=False,
    )


class ShellDispatchTests(unittest.TestCase):
    def _server(self, opened, transcripts=None):
        def open_shell(label):
            opened.append(label)
            session = mock.Mock(spec=podman_host.ContainerShellSession)
            session.exec.return_value = _result(out=f"on {label}\n")
            return session, (transcripts or {}).get(label, "")

        server = shell_socket.ShellDispatchServer(
            machine_labels=("x86_64", "arm64"),
            open_shell=open_shell,
            max_timeout_s=60.0,
        )
        self.addCleanup(server.close)
        return server

    def test_round_trip_and_machine_routing(self) -> None:
        opened = []
        server = self._server(opened)
        client = shell_socket.ShellDispatchClient(server.socket_path)
        self.addCleanup(client.close)
        payload = client.call({"command": "true"})
        self.assertEqual("on x86_64\n", payload["stdout"])
        payload = client.call({"command": "true", "machine": "arm64"})
        self.assertEqual("on arm64\n", payload["stdout"])
        self.assertEqual(["x86_64", "arm64"], opened)

    def test_sessions_are_reused_within_a_connection(self) -> None:
        opened = []
        server = self._server(opened)
        client = shell_socket.ShellDispatchClient(server.socket_path)
        self.addCleanup(client.close)
        client.call({"command": "a"})
        client.call({"command": "b"})
        self.assertEqual(["x86_64"], opened)

    def test_connections_get_isolated_container_sets(self) -> None:
        # One connection = one codex run; concurrent runs must not share
        # a working tree, so each connection opens its own containers.
        opened = []
        server = self._server(opened)
        for _ in range(2):
            client = shell_socket.ShellDispatchClient(server.socket_path)
            self.addCleanup(client.close)
            client.call({"command": "true"})
        self.assertEqual(["x86_64", "x86_64"], opened)

    def test_unknown_machine_is_in_band_error(self) -> None:
        server = self._server([])
        client = shell_socket.ShellDispatchClient(server.socket_path)
        self.addCleanup(client.close)
        payload = client.call({"command": "true", "machine": "riscv"})
        self.assertIn("unknown machine", payload["error"])

    def test_failed_open_is_in_band_error(self) -> None:
        def open_shell(label):
            raise RuntimeError("provisioning exploded")

        server = shell_socket.ShellDispatchServer(
            machine_labels=("x86_64",), open_shell=open_shell,
            max_timeout_s=60.0,
        )
        self.addCleanup(server.close)
        client = shell_socket.ShellDispatchClient(server.socket_path)
        self.addCleanup(client.close)
        payload = client.call({"command": "true"})
        self.assertIn("provisioning exploded", payload["error"])

    def test_setup_transcript_rides_first_non_default_result(self) -> None:
        opened = []
        server = self._server(opened, transcripts={"arm64": "$ git status\n"})
        client = shell_socket.ShellDispatchClient(server.socket_path)
        self.addCleanup(client.close)
        first = client.call({"command": "a", "machine": "arm64"})
        self.assertEqual("$ git status\n", first["setup_transcript"])
        second = client.call({"command": "b", "machine": "arm64"})
        self.assertNotIn("setup_transcript", second)

    def test_close_removes_socket_and_directory(self) -> None:
        server = self._server([])
        path = server.socket_path
        self.assertTrue(os.path.exists(path))
        server.close()
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(os.path.dirname(path)))

    def test_socket_directory_is_private(self) -> None:
        server = self._server([])
        mode = os.stat(os.path.dirname(server.socket_path)).st_mode & 0o777
        self.assertEqual(0o700, mode)

    def test_concurrent_connections_do_not_interleave(self) -> None:
        opened = []
        server = self._server(opened)
        results = []

        def one_client():
            client = shell_socket.ShellDispatchClient(server.socket_path)
            try:
                for _ in range(5):
                    results.append(client.call({"command": "true"})["stdout"])
            finally:
                client.close()

        threads = [threading.Thread(target=one_client) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(["on x86_64\n"] * 15, results)
        self.assertEqual(["x86_64"] * 3, opened)


class RelayDispatchTests(unittest.TestCase):
    """The codex-container transport: codex_bridge -> relay.py socket ->
    relay stdio -> serve_dispatch (the wrapper side), no bind mount / no
    reverse tunnel. Drives the real relay.py as a subprocess, exactly as
    the wrapper will over ``podman exec -i``."""

    def _open_shell(self, opened, transcripts=None):
        def open_shell(label):
            opened.append(label)
            session = mock.Mock(spec=podman_host.ContainerShellSession)
            session.exec.return_value = _result(out=f"on {label}\n")
            return session, (transcripts or {}).get(label, "")
        return open_shell

    def _start_relay(self, sock_path):
        proc = subprocess.Popen(
            [sys.executable, str(RELAY_PY), sock_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        def _cleanup():
            proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
        self.addCleanup(_cleanup)
        self.assertIn(b"RELAY-READY", proc.stderr.readline())
        return proc

    def test_round_trip_and_routing_through_relay(self) -> None:
        sock = os.path.join(tempfile.mkdtemp(prefix="relay-test-"), "shell.sock")
        proc = self._start_relay(sock)
        opened = []
        threading.Thread(
            target=shell_socket.serve_dispatch,
            args=(proc.stdout, proc.stdin),
            kwargs=dict(machine_labels=("x86_64", "arm64"),
                        open_shell=self._open_shell(opened), max_timeout_s=60.0),
            daemon=True,
        ).start()

        client = shell_socket.ShellDispatchClient(sock)
        self.addCleanup(client.close)
        self.assertEqual("on x86_64\n", client.call({"command": "true"})["stdout"])
        self.assertEqual(
            "on arm64\n",
            client.call({"command": "true", "machine": "arm64"})["stdout"],
        )
        self.assertEqual(["x86_64", "arm64"], opened)

    def test_unknown_machine_is_in_band_error_through_relay(self) -> None:
        sock = os.path.join(tempfile.mkdtemp(prefix="relay-test-"), "shell.sock")
        proc = self._start_relay(sock)
        threading.Thread(
            target=shell_socket.serve_dispatch,
            args=(proc.stdout, proc.stdin),
            kwargs=dict(machine_labels=("x86_64",),
                        open_shell=self._open_shell([]), max_timeout_s=60.0),
            daemon=True,
        ).start()
        client = shell_socket.ShellDispatchClient(sock)
        self.addCleanup(client.close)
        payload = client.call({"command": "true", "machine": "riscv"})
        self.assertIn("unknown machine", payload["error"])


if __name__ == "__main__":
    unittest.main()
