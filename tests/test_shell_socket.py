"""Shell dispatch over the per-run unix socket (codex backend plumbing)."""

import json
import os
import socket
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


class ServeDispatchTests(unittest.TestCase):
    """serve_dispatch over an in-memory socket pair -- the same dispatch
    loop the relay drives, without the podman-exec subprocess."""

    def _serve(self, opened, transcripts=None, machine_labels=("x86_64", "arm64")):
        def open_shell(label):
            opened.append(label)
            session = mock.Mock(spec=podman_host.ContainerShellSession)
            session.exec.return_value = _result(out=f"on {label}\n")
            return session, (transcripts or {}).get(label, "")

        srv, cli = socket.socketpair()
        self.addCleanup(srv.close)
        self.addCleanup(cli.close)
        threading.Thread(
            target=shell_socket.serve_dispatch,
            args=(srv.makefile("rb"), srv.makefile("wb")),
            kwargs=dict(machine_labels=machine_labels, open_shell=open_shell,
                        max_timeout_s=60.0),
            daemon=True,
        ).start()
        self._cw = cli.makefile("wb")
        self._cr = cli.makefile("rb")
        return self

    def _call(self, args):
        self._cw.write(json.dumps({"id": 1, "args": args}).encode() + b"\n")
        self._cw.flush()
        return json.loads(self._cr.readline())["payload"]

    def test_round_trip_and_machine_routing(self) -> None:
        opened = []
        self._serve(opened)
        self.assertEqual("on x86_64\n", self._call({"command": "true"})["stdout"])
        self.assertEqual(
            "on arm64\n",
            self._call({"command": "true", "machine": "arm64"})["stdout"])
        self.assertEqual(["x86_64", "arm64"], opened)

    def test_sessions_are_reused_within_a_run(self) -> None:
        opened = []
        self._serve(opened)
        self._call({"command": "a"})
        self._call({"command": "b"})
        self.assertEqual(["x86_64"], opened)  # one dict per serve_dispatch

    def test_unknown_machine_is_in_band_error(self) -> None:
        self._serve([])
        payload = self._call({"command": "true", "machine": "riscv"})
        self.assertIn("unknown machine", payload["error"])

    def test_failed_open_is_in_band_error(self) -> None:
        def open_shell(label):
            raise RuntimeError("provisioning exploded")

        srv, cli = socket.socketpair()
        self.addCleanup(srv.close)
        self.addCleanup(cli.close)
        threading.Thread(
            target=shell_socket.serve_dispatch,
            args=(srv.makefile("rb"), srv.makefile("wb")),
            kwargs=dict(machine_labels=("x86_64",), open_shell=open_shell,
                        max_timeout_s=60.0),
            daemon=True,
        ).start()
        self._cw = cli.makefile("wb")
        self._cr = cli.makefile("rb")
        payload = self._call({"command": "true"})
        self.assertIn("provisioning exploded", payload["error"])

    def test_setup_transcript_rides_first_non_default_result(self) -> None:
        opened = []
        self._serve(opened, transcripts={"arm64": "$ git status\n"})
        first = self._call({"command": "a", "machine": "arm64"})
        self.assertEqual("$ git status\n", first["setup_transcript"])
        second = self._call({"command": "b", "machine": "arm64"})
        self.assertNotIn("setup_transcript", second)

    def test_shells_arg_exposes_opened_sessions(self) -> None:
        shells = {}
        opened = []

        def open_shell(label):
            opened.append(label)
            session = mock.Mock(spec=podman_host.ContainerShellSession)
            session.exec.return_value = _result(out="ok\n")
            return session, ""

        srv, cli = socket.socketpair()
        self.addCleanup(srv.close)
        self.addCleanup(cli.close)
        threading.Thread(
            target=shell_socket.serve_dispatch,
            args=(srv.makefile("rb"), srv.makefile("wb")),
            kwargs=dict(machine_labels=("x86_64",), open_shell=open_shell,
                        max_timeout_s=60.0, shells=shells),
            daemon=True,
        ).start()
        cw = cli.makefile("wb")
        cr = cli.makefile("rb")
        cw.write(json.dumps({"id": 1, "args": {"command": "true"}}).encode() + b"\n")
        cw.flush()
        cr.readline()
        self.assertEqual(["x86_64"], list(shells))


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
