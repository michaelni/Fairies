"""Shell dispatch over the per-run unix socket (codex backend plumbing)."""

import os
import threading
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import shell_socket


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


if __name__ == "__main__":
    unittest.main()
