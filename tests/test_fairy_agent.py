"""Tests for containers/fairy_agent.py.

These spawn the real agent as a subprocess and drive the actual NDJSON
protocol over its stdin/stdout -- no container or podman needed, because
the agent is plain stdlib Python. This is exactly how the in-container
agent is exercised in production, so it doubles as the regression test
for the shell-execution contract (exit codes, base64 round-trip,
watchdog timeout killing the process group, output capping).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent / "containers" / "fairy_agent.py"


class AgentProc:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(AGENT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def request(self, **fields) -> dict:
        self.proc.stdin.write((json.dumps(fields) + "\n").encode())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError("agent closed stdout unexpectedly")
        return json.loads(line)

    def send_raw(self, raw: bytes) -> dict:
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        self.proc.stdout.close()
        self.proc.stderr.close()


def _stdout(resp: dict) -> bytes:
    return base64.b64decode(resp["stdout_b64"])


def _stderr(resp: dict) -> bytes:
    return base64.b64decode(resp["stderr_b64"])


class FairyAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AgentProc()
        self.addCleanup(self.agent.close)

    def test_echo_stdout_and_exit_zero(self) -> None:
        r = self.agent.request(id=1, command="echo hello")
        self.assertEqual(1, r["id"])
        self.assertEqual(0, r["exit_code"])
        self.assertEqual(b"hello\n", _stdout(r))

    def test_exit_code_propagates(self) -> None:
        self.assertEqual(3, self.agent.request(id=2, command="exit 3")["exit_code"])

    def test_stderr_captured_separately(self) -> None:
        r = self.agent.request(id=3, command="echo oops >&2")
        self.assertEqual(b"", _stdout(r))
        self.assertEqual(b"oops\n", _stderr(r))

    def test_cwd_is_honored(self) -> None:
        r = self.agent.request(id=4, command="pwd", cwd="/tmp")
        self.assertEqual(b"/tmp\n", _stdout(r))

    def test_binary_output_survives_base64(self) -> None:
        r = self.agent.request(id=5, command=r"printf '\x00\x01\xff\x0a'")
        self.assertEqual(b"\x00\x01\xff\x0a", _stdout(r))

    def test_timeout_kills_and_returns_124(self) -> None:
        t0 = time.monotonic()
        r = self.agent.request(id=6, command="sleep 30", timeout_s=0.5)
        self.assertEqual(124, r["exit_code"])
        self.assertLess(time.monotonic() - t0, 15.0)

    def test_timeout_kills_whole_process_group(self) -> None:
        # The command backgrounds a child; the watchdog must kill the
        # whole group, not just bash, so nothing is left running.
        marker = "/tmp/fairy_agent_pg_test.marker"
        subprocess.run(["rm", "-f", marker])
        r = self.agent.request(
            id=7, timeout_s=0.5,
            command=f"(sleep 3; touch {marker}) & sleep 30",
        )
        self.assertEqual(124, r["exit_code"])
        time.sleep(4)
        self.assertFalse(Path(marker).exists(),
                         "backgrounded child survived the group kill")

    def test_output_capping_sets_truncated(self) -> None:
        r = self.agent.request(
            id=8, command="yes ABCDEFGH | head -c 100000", max_output_bytes=1024,
        )
        self.assertTrue(r["stdout_truncated"])
        self.assertLessEqual(len(_stdout(r)), 1024)

    def test_bad_json_keeps_stream_usable(self) -> None:
        bad = self.agent.send_raw(b"{not json}\n")
        self.assertEqual(127, bad["exit_code"])
        self.assertIsNone(bad["id"])
        good = self.agent.request(id=9, command="echo still-here")
        self.assertEqual(b"still-here\n", _stdout(good))


if __name__ == "__main__":
    unittest.main()
