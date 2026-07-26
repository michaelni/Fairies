"""workset_to_filedb: the migrated wait must equal the old window."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent  # noqa: E402


class MigrationBackoffTests(unittest.TestCase):
    def test_skip_counter_n_migrates_to_the_same_wait_as_before(self) -> None:
        # old gate: window after n consecutive skips = 24 * 2**(n-1) h;
        # new gate: wait = max(24, 2 * skip_backoff_h). The migrated
        # value must make both sides of that equation agree, or every
        # parked item waits double (cutover A/B finding, 2026-07-25).
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name) / "src"
        src.mkdir()
        for n, number in ((1, 1), (2, 2), (5, 3)):
            (src / f"pr-{number}.json").write_text(json.dumps({
                "state": 5, "review": {"classification": "skip",
                                       "message": "", "label_changes": []},
                "consecutive_skip_count": n, "llm_at": "x"}))
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools/workset_to_filedb.py"),
             str(src), str(Path(tmp.name) / "db")], check=True,
            capture_output=True)
        for n, number in ((1, 1), (2, 2), (5, 3)):
            t = json.loads(
                (Path(tmp.name) / f"db/skipped/pr-{number}.json").read_text())
            self.assertEqual(agent.backoff_wait_h(t["skip_backoff_h"]),
                             24.0 * 2 ** (n - 1),
                             f"counter {n}: wait differs from the old window")


class MigrationInFlightTests(unittest.TestCase):
    def test_in_flight_items_become_rerun_requests(self) -> None:
        # old in-flight files carry no prepared payload; a queued/
        # ticket would crash the worker into error/ and stall 24h --
        # a request makes the agent rebuild a fresh payload instead
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name) / "src"
        src.mkdir()
        for state, number in ((1, 1), (3, 2)):
            (src / f"pr-{number}.json").write_text(json.dumps({
                "state": state, "title": "t"}))
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools/workset_to_filedb.py"),
             str(src), str(Path(tmp.name) / "db")], check=True,
            capture_output=True)
        for number in (1, 2):
            t = json.loads((Path(tmp.name)
                            / f"db/requests/pr-{number}.json").read_text())
            self.assertEqual(t["action"], "rerun")


class MigrationActionTests(unittest.TestCase):
    def test_reviewed_verdicts_gain_the_action_field(self) -> None:
        # log_summary and the TUI awaiting-you stats key on it
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name) / "src"
        src.mkdir()
        (src / "pr-1.json").write_text(json.dumps({
            "state": 5, "review": {"classification": "major_issues",
                                   "message": "m", "label_changes": []}}))
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools/workset_to_filedb.py"),
             str(src), str(Path(tmp.name) / "db")], check=True,
            capture_output=True)
        t = json.loads((Path(tmp.name) / "db/reviewed/pr-1.json").read_text())
        self.assertEqual(t["action"], "request_changes")


if __name__ == "__main__":
    unittest.main()
