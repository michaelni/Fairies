"""filedb: atomic state-as-directory operations, claim/reap races."""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import filedb  # noqa: E402


class DbCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))


class HygieneTests(DbCase):
    def age(self, state: str, kind: str, number: int) -> None:
        data = self.db.get(state, kind, number)
        data["state_changed_at"] = (
            datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self.db._write(self.db.path(state, kind, number), data)

    def cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=14)

    def test_prune_drops_the_lock_file_with_the_last_trace(self) -> None:
        self.db.push("posted", "pr", 5, {"title": "t"})
        self.age("posted", "pr", 5)
        self.assertTrue(self.db._lock_path("pr", 5).exists())
        self.db.prune("posted", self.cutoff())
        self.assertFalse(self.db._lock_path("pr", 5).exists())
        # the item can come back: locking must still work on a fresh file
        self.db.push("queued", "pr", 5, {"title": "again"})
        self.assertEqual(self.db.find("pr", 5), "queued")

    def test_prune_keeps_the_lock_while_another_state_remains(self) -> None:
        self.db.push("posted", "pr", 5, {"title": "t"})
        self.db.push("skipped", "pr", 5, {"llm_at": "x"})
        self.age("posted", "pr", 5)
        self.db.prune("posted", self.cutoff())
        self.assertTrue(self.db._lock_path("pr", 5).exists())

    def test_reap_cleans_the_dead_workers_sidecar_litter(self) -> None:
        self.db.push("llm", "pr", 5, {"title": "t"})
        path = self.db.path("llm", "pr", 5)
        path.with_suffix(".lock").write_text("")  # workset.update_json's
        path.with_suffix(".tmp").write_text("{}")
        self.assertEqual(self.db.reap(), [("pr", 5)])
        self.assertFalse(path.with_suffix(".lock").exists())
        self.assertFalse(path.with_suffix(".tmp").exists())


class TryPopTests(DbCase):
    def test_try_pop_refuses_claimed_items_instead_of_blocking(self) -> None:
        self.db.push("requests", "pr", 5, {"action": "rerun"})
        claim = self.db.claim("requests", "requests", "pr", 5)
        try:
            self.assertIsNone(self.db.try_pop("requests", "pr", 5))
        finally:
            claim.abort()
        self.assertEqual(self.db.try_pop("requests", "pr", 5)["action"],
                         "rerun")
        self.assertIsNone(self.db.get("requests", "pr", 5))


class ReplaceTests(DbCase):
    def test_replace_moves_wherever_the_caller_expected(self) -> None:
        self.db.push("reviewed", "pr", 5, {"title": "old"})
        self.assertTrue(self.db.replace("queued", "pr", 5, {"title": "new"},
                                        expect="reviewed"))
        self.assertEqual(self.db.find("pr", 5), "queued")
        self.assertIsNone(self.db.get("reviewed", "pr", 5))
        self.assertEqual(self.db.get("queued", "pr", 5)["title"], "new")

    def test_replace_refuses_moved_and_claimed_items(self) -> None:
        # the item moved under the caller (here: reviewed -> outgoing,
        # an operator y): the decision made against "reviewed" is stale
        self.db.push("outgoing", "pr", 5, {"title": "pending send"})
        self.assertFalse(self.db.replace("queued", "pr", 5, {},
                                         expect="reviewed"))
        self.assertEqual(self.db.get("outgoing", "pr", 5)["title"],
                         "pending send")
        self.db.push("reviewed", "pr", 6, {"title": "t"})
        claim = self.db.claim("reviewed", "reviewed", "pr", 6)
        try:
            self.assertFalse(self.db.replace("queued", "pr", 6, {},
                                             expect="reviewed"))
        finally:
            claim.abort()
        self.assertEqual(self.db.find("pr", 6), "reviewed")


class BasicOpsTests(DbCase):
    def test_push_get_pop_roundtrip(self) -> None:
        self.db.push("queued", "pr", 5, {"title": "t"})
        data = self.db.get("queued", "pr", 5)
        self.assertEqual(data["title"], "t")
        self.assertIn("state_changed_at", data)
        self.assertEqual(self.db.find("pr", 5), "queued")
        self.assertEqual(self.db.pop("queued", "pr", 5), data)
        self.assertIsNone(self.db.get("queued", "pr", 5))
        self.assertIsNone(self.db.pop("queued", "pr", 5))

    def test_move_mutates_and_returns_false_when_racing(self) -> None:
        self.db.push("skipped", "pr", 7, {"skip_backoff": 2})
        moved = self.db.move("skipped", "queued", "pr", 7,
                             mutate=lambda d: d.update(
                                 skip_backoff=d["skip_backoff"] * 2))
        self.assertTrue(moved)
        self.assertEqual(self.db.get("queued", "pr", 7)["skip_backoff"], 4)
        self.assertIsNone(self.db.get("skipped", "pr", 7))
        self.assertFalse(self.db.move("skipped", "queued", "pr", 7))

    def test_list_state_and_kinds(self) -> None:
        self.db.push("queued", "pr", 2, {})
        self.db.push("queued", "issue", 2, {})
        self.db.push("reviewed", "pr", 9, {})
        self.assertEqual(self.db.list_state("queued"),
                         [("issue", 2), ("pr", 2)])
        self.assertEqual(self.db.list_state("reviewed"), [("pr", 9)])

    def test_unknown_state_and_kind_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.db.push("nonsense", "pr", 1, {})
        with self.assertRaises(ValueError):
            self.db.push("queued", "branch", 1, {})


def _claim_once(root: str, results, barrier) -> None:
    db = filedb.Db(Path(root))
    barrier.wait()
    claim = db.claim("queued", "llm", "pr", 5)
    if claim is not None:
        time.sleep(0.2)  # hold the claim while the losers report
        claim.finish("reviewed", claim.read())
        results.put(os.getpid())


def _claim_and_hang(root: str, ready) -> None:
    db = filedb.Db(Path(root))
    claim = db.claim("queued", "llm", "pr", 5)
    ready.put(claim is not None)
    time.sleep(60)  # hold the flock until killed


class ClaimRaceTests(DbCase):
    def test_exactly_one_of_many_claimants_wins(self) -> None:
        self.db.push("queued", "pr", 5, {"title": "t"})
        results: multiprocessing.Queue = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(6)
        procs = [multiprocessing.Process(
            target=_claim_once, args=(str(self.db.root), results, barrier))
            for _ in range(6)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
        winners = []
        while not results.empty():
            winners.append(results.get())
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.db.find("pr", 5), "reviewed")

    def test_finish_releases_the_lock(self) -> None:
        self.db.push("queued", "pr", 5, {})
        claim = self.db.claim("queued", "llm", "pr", 5)
        claim.finish("reviewed", {"verdict": "x"})
        self.assertEqual(self.db.get("reviewed", "pr", 5)["verdict"], "x")
        self.assertIsNone(self.db.get("llm", "pr", 5))
        # the item is claimable again (from its new state)
        again = self.db.claim("reviewed", "outgoing", "pr", 5)
        self.assertIsNotNone(again)
        again.abort()
        self.assertEqual(self.db.find("pr", 5), "reviewed")

    def test_abort_returns_the_ticket(self) -> None:
        self.db.push("queued", "pr", 5, {})
        claim = self.db.claim("queued", "llm", "pr", 5)
        claim.abort()
        self.assertEqual(self.db.find("pr", 5), "queued")
        self.assertIsNotNone(self.db.claim("queued", "llm", "pr", 5))


class ReaperTests(DbCase):
    def test_live_claim_is_not_reaped_dead_one_is(self) -> None:
        self.db.push("queued", "pr", 5, {"title": "t"})
        ready: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_claim_and_hang, args=(str(self.db.root), ready))
        p.start()
        self.assertTrue(ready.get(timeout=10))
        self.assertEqual(self.db.reap(), [])          # holder alive
        self.assertEqual(self.db.find("pr", 5), "llm")
        os.kill(p.pid, signal.SIGKILL)
        p.join(timeout=10)
        self.assertEqual(self.db.reap(), [("pr", 5)])  # flock died with it
        self.assertEqual(self.db.find("pr", 5), "queued")
        self.assertEqual(self.db.get("queued", "pr", 5)["title"], "t")

    def test_crash_remnant_in_earlier_state_is_deleted(self) -> None:
        # A crash between the reviewed/ write and the llm/ unlink leaves
        # both files; the later pipeline state wins.
        self.db.push("llm", "pr", 5, {"old": True})
        self.db.push("reviewed", "pr", 5, {"old": False})
        self.assertEqual(self.db.find("pr", 5), "reviewed")
        self.assertEqual(self.db.reap(), [])
        self.assertIsNone(self.db.get("llm", "pr", 5))
        self.assertEqual(self.db.get("reviewed", "pr", 5)["old"], False)


class PruneTests(DbCase):
    def test_prune_respects_age_and_keep(self) -> None:
        self.db.push("posted", "pr", 1, {})
        self.db.push("posted", "pr", 2, {})
        self.db.push("skipped", "pr", 3, {})
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        for kind, num, state in (("pr", 1, "posted"), ("pr", 3, "skipped")):
            data = self.db.get(state, kind, num)
            data["state_changed_at"] = old
            self.db._write(self.db.path(state, kind, num), data)
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)
        self.assertEqual(self.db.prune("posted", cutoff), 1)
        # pr-3 is old but kept: its skip verdict is backoff memory
        self.assertEqual(self.db.prune("skipped", cutoff, keep={("pr", 3)}), 0)
        self.assertIsNone(self.db.get("posted", "pr", 1))
        self.assertIsNotNone(self.db.get("posted", "pr", 2))
        self.assertIsNotNone(self.db.get("skipped", "pr", 3))


if __name__ == "__main__":
    unittest.main()
