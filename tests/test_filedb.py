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

filedb: atomic state-as-directory operations, claim/reap races."""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import tempfile
import threading
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

    def test_reap_cleans_the_dead_workers_sidecar_litter(self) -> None:
        self.db.push("llm", "pr", "5", {"title": "t"})
        path = self.db.path("llm", "pr", "5")
        path.with_suffix(".lock").write_text("")  # workset.update_json's
        path.with_suffix(".tmp").write_text("{}")
        self.assertEqual(self.db.reap(), [("pr", "5")])
        self.assertFalse(path.with_suffix(".lock").exists())
        self.assertFalse(path.with_suffix(".tmp").exists())


class ReapRemnantOnlyTests(DbCase):
    def test_to_state_none_deletes_remnants_but_never_requeues(self) -> None:
        self.db.push("reviewed", "pr", "1", {"title": "lone: valid"})
        self.db.push("reviewed", "pr", "2", {"title": "remnant"})
        self.db.push("outgoing", "pr", "2", {"title": "real"})
        self.assertEqual(self.db.reap("reviewed", None), [])
        self.assertIsNotNone(self.db.get("reviewed", "pr", "1"))
        self.assertIsNone(self.db.get("reviewed", "pr", "2"))
        self.assertIsNotNone(self.db.get("outgoing", "pr", "2"))

    def test_reap_waits_for_a_transition_in_progress(self) -> None:
        """replace() writes dst and only then unlinks src, both under the
        transition lock; a reap in that window sees the two files of a
        crash remnant and would delete the freshly written one."""
        self.db.push("skipped", "pr", "5", {"t": "old"})
        mid_transition = threading.Event()

        def backward_move() -> None:
            with self.db.lock("pr", "5"):
                self.db._write_state("queued", "pr", "5", {"t": "new"})
                mid_transition.set()
                time.sleep(0.3)
                self.db.path("skipped", "pr", "5").unlink()

        writer = threading.Thread(target=backward_move)
        writer.start()
        try:
            self.assertTrue(mid_transition.wait(10))
            self.db.reap("queued", None)
        finally:
            writer.join(timeout=10)
        self.assertTrue(self.db.path("queued", "pr", "5").exists())
        self.assertIsNone(self.db.get("skipped", "pr", "5"))


class TokenTests(DbCase):
    def test_sample_and_review_tokens_coexist_with_the_base(self) -> None:
        for token in ("12345", "12345s1", "12345s2", "12345s1r2"):
            self.db.push("queued", "pr", token, {"eval": str(token)})
        self.assertEqual(self.db.list_state("queued"),
                         [("pr", "12345"), ("pr", "12345s1"),
                          ("pr", "12345s1r2"), ("pr", "12345s2")])
        self.assertEqual(self.db.get("queued", "pr", "12345s2")["eval"],
                         "12345s2")

    def test_forge_number_and_is_base(self) -> None:
        self.assertEqual(filedb.forge_number("12345s1r2"), 12345)
        self.assertEqual(filedb.forge_number("12345"), 12345)
        self.assertTrue(filedb.is_base("12345"))
        self.assertFalse(filedb.is_base("12345s1"))
        with self.assertRaises(ValueError):
            self.db.push("queued", "pr", "12345x9", {})


class NoOpWriteTests(DbCase):
    def test_identical_content_is_not_rewritten(self) -> None:
        # every writer benefits: no fsync churn, no dir-mtime bump for
        # viewers, state_changed_at keeps meaning "last actual change"
        self.db.push("ci-blocked", "pr", "5", {"reason": "ci red"})
        stamp = self.db.path("ci-blocked", "pr", "5").stat().st_mtime_ns
        self.db.push("ci-blocked", "pr", "5", {"reason": "ci red"})
        self.assertEqual(
            self.db.path("ci-blocked", "pr", "5").stat().st_mtime_ns, stamp)
        self.db.push("ci-blocked", "pr", "5", {"reason": "ci red, job2"})
        self.assertNotEqual(
            self.db.path("ci-blocked", "pr", "5").stat().st_mtime_ns, stamp)


class ItemSnapshotTests(DbCase):
    SNAP = {"title": "t", "author": "a", "body": "b",
            "discussion": [{"kind": "comment", "author": "a", "body": "hi"}]}

    def test_the_item_state_is_ordinary_but_outside_the_pipeline(self) -> None:
        self.db.push("items", "pr", "5", self.SNAP)
        self.assertEqual(self.db.get("items", "pr", "5")["title"], "t")
        self.assertEqual(self.db.list_state("items"), [("pr", "5")])
        self.assertIsNone(self.db.find("pr", "5"))
        self.assertEqual(self.db.list_state("queued"), [])

    def test_a_snapshot_coexists_with_a_claimed_ticket(self) -> None:
        """A claim is about a ticket's review; the same item's snapshot
        must stay writable while the review runs."""
        self.db.push("queued", "pr", "5", {"title": "t"})
        held = self.db.claim("queued", "llm", "pr", "5")
        self.assertIsNotNone(held)
        self.db.push("items", "pr", "5", self.SNAP)
        self.assertEqual(self.db.get("items", "pr", "5")["discussion"],
                         self.SNAP["discussion"])
        held.abort()


class TornTicketTests(DbCase):
    def test_torn_tickets_degrade_instead_of_wedging(self) -> None:
        self.db.push("skipped", "pr", "5", {"a": 1})
        self.db.path("skipped", "pr", "5").write_text("{ torn")
        self.assertIsNone(self.db.get("skipped", "pr", "5"))
        self.assertIsNone(self.db.try_pop("skipped", "pr", "5"))


class TryPopTests(DbCase):
    def test_try_pop_refuses_claimed_items_but_not_requests(self) -> None:
        """The requests exemption is production pr #23914: a request
        that cannot be consumed while its review runs re-forces the
        item every pass and destroys each fresh verdict."""
        self.db.push("queued", "pr", "5", {"title": "t"})
        claim = self.db.claim("queued", "llm", "pr", "5")
        try:
            self.assertIsNone(self.db.try_pop("llm", "pr", "5"))
            self.db.request("pr", "5", {"action": "rerun"})
            self.assertEqual(self.db.try_pop("requests", "pr", "5")["action"],
                             "rerun")
        finally:
            claim.abort()
        self.assertIsNone(self.db.get("requests", "pr", "5"))


class ReplaceTests(DbCase):
    def test_replace_moves_wherever_the_caller_expected(self) -> None:
        self.db.push("reviewed", "pr", "5", {"title": "old"})
        self.assertTrue(self.db.replace("queued", "pr", "5", {"title": "new"},
                                        expect="reviewed"))
        self.assertEqual(self.db.find("pr", "5"), "queued")
        self.assertIsNone(self.db.get("reviewed", "pr", "5"))
        self.assertEqual(self.db.get("queued", "pr", "5")["title"], "new")

    def test_replace_refuses_moved_and_claimed_items(self) -> None:
        # the item moved under the caller (here: reviewed -> outgoing,
        # an operator y): the decision made against "reviewed" is stale
        self.db.push("outgoing", "pr", "5", {"title": "pending send"})
        self.assertFalse(self.db.replace("queued", "pr", "5", {},
                                         expect="reviewed"))
        self.assertEqual(self.db.get("outgoing", "pr", "5")["title"],
                         "pending send")
        self.db.push("reviewed", "pr", "6", {"title": "t"})
        claim = self.db.claim("reviewed", "reviewed", "pr", "6")
        try:
            self.assertFalse(self.db.replace("queued", "pr", "6", {},
                                             expect="reviewed"))
        finally:
            claim.abort()
        self.assertEqual(self.db.find("pr", "6"), "reviewed")


class LeaseSplitTests(DbCase):
    """A review lease (.claim) and the transition lock (.lock) are
    separate: micro-duration ops must never wait out a review."""

    def test_prune_does_not_block_on_a_live_claim(self) -> None:
        import time as _t
        from datetime import datetime, timedelta, timezone
        self.db.push("posted", "pr", "5", {})
        data = self.db.get("posted", "pr", "5")
        data["state_changed_at"] = (
            datetime.now(timezone.utc) - timedelta(days=99)).isoformat()
        self.db._write(self.db.path("posted", "pr", "5"), data)
        self.db.push("queued", "pr", "5", {})
        claim = self.db.claim("queued", "llm", "pr", "5")
        try:
            start = _t.monotonic()
            removed = self.db.prune(
                "posted", datetime.now(timezone.utc) - timedelta(days=14),
                keep=set())
            self.assertLess(_t.monotonic() - start, 2.0)
            self.assertEqual(removed, 1)
            self.assertIsNone(self.db.try_pop("llm", "pr", "5"))
            self.db.request("pr", "5", {"action": "rerun"})
            self.assertIsNotNone(self.db.try_pop("requests", "pr", "5"))
        finally:
            claim.abort()
        self.assertEqual(self.db.find("pr", "5"), "queued")


class InPlaceClaimTests(DbCase):
    """A same-state claim renames nothing: a no-op rename still fires a
    watcher event, and an agent watching the dir would wake itself in
    an endless loop (hit in production: --dry-run send + y)."""

    def test_claim_and_abort_leave_the_dir_untouched(self) -> None:
        import os as _os
        self.db.push("outgoing", "pr", "5", {"title": "t"})
        d = self.db.root / "outgoing"
        _os.utime(d, (100.0, 100.0))
        stamp = d.stat().st_mtime_ns
        claim = self.db.claim("outgoing", "outgoing", "pr", "5")
        self.assertEqual(claim.read()["title"], "t")
        claim.abort()
        self.assertEqual(d.stat().st_mtime_ns, stamp)
        self.assertEqual(self.db.find("pr", "5"), "outgoing")

    def test_claiming_a_missing_file_in_place_returns_none(self) -> None:
        self.assertIsNone(self.db.claim("outgoing", "outgoing", "pr", "5"))
        self.db.push("outgoing", "pr", "5", {})
        claim = self.db.claim("outgoing", "outgoing", "pr", "5")
        self.assertIsNotNone(claim)
        claim.abort()


class BasicOpsTests(DbCase):
    def test_push_get_roundtrip(self) -> None:
        self.db.push("queued", "pr", "5", {"title": "t"})
        data = self.db.get("queued", "pr", "5")
        self.assertEqual(data["title"], "t")
        self.assertIn("state_changed_at", data)
        self.assertEqual(self.db.find("pr", "5"), "queued")

    def test_the_callers_dict_is_not_stamped(self) -> None:
        data = {"title": "t"}
        self.db.push("queued", "pr", "5", data)
        self.assertEqual(data, {"title": "t"})
        self.assertIn("state_changed_at", self.db.get("queued", "pr", "5"))

    def test_try_move_mutates_and_returns_false_when_racing(self) -> None:
        self.db.push("skipped", "pr", "7", {"skip_backoff": 2})
        moved = self.db.try_move("skipped", "queued", "pr", "7",
                                 mutate=lambda d: d.update(
                                     skip_backoff=d["skip_backoff"] * 2))
        self.assertTrue(moved)
        self.assertEqual(self.db.get("queued", "pr", "7")["skip_backoff"], 4)
        self.assertIsNone(self.db.get("skipped", "pr", "7"))
        self.assertFalse(self.db.try_move("skipped", "queued", "pr", "7"))

    def test_list_state_and_kinds(self) -> None:
        self.db.push("queued", "pr", "2", {})
        self.db.push("queued", "issue", "2", {})
        self.db.push("reviewed", "pr", "9", {})
        self.assertEqual(self.db.list_state("queued"),
                         [("issue", "2"), ("pr", "2")])
        self.assertEqual(self.db.list_state("reviewed"), [("pr", "9")])

    def test_unknown_state_and_kind_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.db.push("nonsense", "pr", "1", {})
        with self.assertRaises(ValueError):
            self.db.push("queued", "branch", "1", {})


def _claim_once(root: str, results, barrier) -> None:
    db = filedb.Db(Path(root))
    barrier.wait()
    claim = db.claim("queued", "llm", "pr", "5")
    if claim is not None:
        time.sleep(0.2)  # hold the claim while the losers report
        claim.finish("reviewed", claim.read())
        results.put(os.getpid())


def _claim_and_hang(root: str, ready) -> None:
    db = filedb.Db(Path(root))
    claim = db.claim("queued", "llm", "pr", "5")
    ready.put(claim is not None)
    time.sleep(60)  # hold the flock until killed


class ClaimRaceTests(DbCase):
    def test_exactly_one_of_many_claimants_wins(self) -> None:
        self.db.push("queued", "pr", "5", {"title": "t"})
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
        self.assertEqual(self.db.find("pr", "5"), "reviewed")

    def test_finish_releases_the_lock(self) -> None:
        self.db.push("queued", "pr", "5", {})
        claim = self.db.claim("queued", "llm", "pr", "5")
        claim.finish("reviewed", {"verdict": "x"})
        self.assertEqual(self.db.get("reviewed", "pr", "5")["verdict"], "x")
        self.assertIsNone(self.db.get("llm", "pr", "5"))
        # the item is claimable again (from its new state)
        again = self.db.claim("reviewed", "outgoing", "pr", "5")
        self.assertIsNotNone(again)
        again.abort()
        self.assertEqual(self.db.find("pr", "5"), "reviewed")

    def test_abort_returns_the_ticket(self) -> None:
        self.db.push("queued", "pr", "5", {})
        claim = self.db.claim("queued", "llm", "pr", "5")
        claim.abort()
        self.assertEqual(self.db.find("pr", "5"), "queued")
        self.assertIsNotNone(self.db.claim("queued", "llm", "pr", "5"))


class ReaperTests(DbCase):
    def test_live_claim_is_not_reaped_dead_one_is(self) -> None:
        self.db.push("queued", "pr", "5", {"title": "t"})
        ready: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_claim_and_hang, args=(str(self.db.root), ready))
        p.start()
        self.assertTrue(ready.get(timeout=10))
        self.assertEqual(self.db.reap(), [])          # holder alive
        self.assertEqual(self.db.find("pr", "5"), "llm")
        os.kill(p.pid, signal.SIGKILL)
        p.join(timeout=10)
        self.assertEqual(self.db.reap(), [("pr", "5")])  # flock died with it
        self.assertEqual(self.db.find("pr", "5"), "queued")
        self.assertEqual(self.db.get("queued", "pr", "5")["title"], "t")

    def test_crash_remnant_in_earlier_state_is_deleted(self) -> None:
        # A crash between the reviewed/ write and the llm/ unlink leaves
        # both files; the later pipeline state wins.
        self.db.push("llm", "pr", "5", {"old": True})
        self.db.push("reviewed", "pr", "5", {"old": False})
        self.assertEqual(self.db.find("pr", "5"), "reviewed")
        self.assertEqual(self.db.reap(), [])
        self.assertIsNone(self.db.get("llm", "pr", "5"))
        self.assertEqual(self.db.get("reviewed", "pr", "5")["old"], False)


class PruneTests(DbCase):
    def test_prune_respects_age_and_keep(self) -> None:
        self.db.push("posted", "pr", "1", {})
        self.db.push("posted", "pr", "2", {})
        self.db.push("skipped", "pr", "3", {})
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        for kind, num, state in (("pr", "1", "posted"), ("pr", "3", "skipped")):
            data = self.db.get(state, kind, num)
            data["state_changed_at"] = old
            self.db._write(self.db.path(state, kind, num), data)
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)
        self.assertEqual(self.db.prune("posted", cutoff), 1)
        # pr-3 is old but kept: its skip verdict is backoff memory
        self.assertEqual(self.db.prune("skipped", cutoff, keep={("pr", "3")}), 0)
        self.assertIsNone(self.db.get("posted", "pr", "1"))
        self.assertIsNotNone(self.db.get("posted", "pr", "2"))
        self.assertIsNotNone(self.db.get("skipped", "pr", "3"))


if __name__ == "__main__":
    unittest.main()
