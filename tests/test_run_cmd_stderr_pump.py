"""Regression test for ``run_cmd``'s ``stderr_line_prefix`` mode.

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

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
