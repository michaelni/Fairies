"""worker: claims become verdict tickets in the right directories."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import filedb  # noqa: E402
import worker  # noqa: E402
import workset  # noqa: E402


def make_pr(n: int) -> dict:
    return {"number": n, "title": f"t{n}", "user": {"login": "a"},
            "updated_at": "2026-07-19T10:00:00Z", "head": {"sha": f"h{n}"},
            "html_url": f"https://forge/pr/{n}"}


def queued_ticket(n: int, backoff: float = 0.0) -> dict:
    prepared = fairy.PreparedPR(
        pr=make_pr(n), number=n, title=f"t{n}", author="a", auto_merge="-",
        last_activity=None, base_reason="review", discussion=[],
        reviewer_username="fairy")
    return {"title": f"t{n}", "author": "a", "skip_backoff_h": backoff,
            "forced": False, "prepared": fairy.prepared_to_dict(prepared)}


def decision(n: int, action: str = "comment", llm: str = "moderate_issues",
             msg: str = "m", labels: tuple = ()) -> fairy.Decision:
    return fairy.Decision(n, f"t{n}", "a", "-", action, "llm", None, llm, msg,
                          label_changes=labels)


class WorkerCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))
        self.ns = fairy.parse_args(["--owner", "o", "--repo", "r"])

    def run_one(self, n: int, result) -> str:
        claim = self.db.claim("queued", "llm", "pr", n)
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=result):
            return worker.review_claim(claim, self.ns)


class VerdictRoutingTests(WorkerCase):
    def test_actionable_review_lands_in_reviewed(self) -> None:
        self.db.push("queued", "pr", 5, queued_ticket(5))
        state = self.run_one(5, lambda ns, p: decision(5))
        self.assertEqual(state, "reviewed")
        t = self.db.get("reviewed", "pr", 5)
        self.assertEqual(t["review"]["classification"], "moderate_issues")
        self.assertEqual(t["expected_updated_at"], "2026-07-19T10:00:00Z")
        self.assertEqual(t["expected_head_ref"], "h5")
        self.assertNotIn("prepared", t)  # the payload is spent
        self.assertIsNone(self.db.get("llm", "pr", 5))

    def test_llm_skip_keeps_its_backoff_in_skipped(self) -> None:
        self.db.push("queued", "pr", 5, queued_ticket(5, backoff=48))
        state = self.run_one(5, lambda ns, p: decision(5, action="skip",
                                                       llm="skip", msg=""))
        self.assertEqual(state, "skipped")
        t = self.db.get("skipped", "pr", 5)
        self.assertEqual(t["skip_backoff_h"], 48)
        self.assertTrue(t["llm_at"])

    def test_skip_with_label_changes_is_operator_actionable(self) -> None:
        self.db.push("queued", "pr", 5, queued_ticket(5))
        labels = (fairy.LabelChange("needs docs", "add", "", False),)
        state = self.run_one(5, lambda ns, p: decision(5, action="skip",
                                                       llm="skip", labels=labels))
        self.assertEqual(state, "reviewed")

    def test_error_lands_in_error(self) -> None:
        self.db.push("queued", "pr", 5, queued_ticket(5))
        state = self.run_one(5, lambda ns, p: decision(5, action="error",
                                                       llm="error", msg=""))
        self.assertEqual(state, "error")
        t = self.db.get("error", "pr", 5)
        self.assertEqual(t["error"], "llm")
        # the guard makes an operator x on the error row stick
        self.assertEqual(t["expected_updated_at"], "2026-07-19T10:00:00Z")

    def test_wrapper_stage_notes_reach_the_claimed_ticket(self) -> None:
        self.db.push("queued", "pr", 5, queued_ticket(5))

        def fake_llm(ns, prepared):
            # the wrapper writes its progress through the override path
            workset.update_json(Path(ns.workset_file_override),
                                lambda d: d.__setitem__("stage", "review"))
            return decision(5)

        self.run_one(5, fake_llm)
        t = self.db.get("reviewed", "pr", 5)
        self.assertNotIn("stage", t)  # transient progress, spent with the run


class DrainTests(WorkerCase):
    def test_drain_reviews_every_queued_ticket_of_its_kinds(self) -> None:
        for n in (1, 2):
            self.db.push("queued", "pr", n, queued_ticket(n))
        self.db.push("queued", "issue", 3, {"title": "i"})  # no issue side
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=lambda ns, p: decision(p.number)):
            done = worker.drain(self.db, {"pr": self.ns})
        self.assertEqual(done, 2)
        self.assertEqual(self.db.list_state("reviewed"), [("pr", 1), ("pr", 2)])
        self.assertEqual(self.db.list_state("queued"), [("issue", 3)])

    def test_parallel_drain_reviews_concurrently_with_isolated_ns(self) -> None:
        # Three tickets, three threads: each review must see its OWN
        # workset_file_override -- a shared namespace would send one
        # ticket's wrapper notes into another ticket's file.
        import threading
        for n in (1, 2, 3):
            self.db.push("queued", "pr", n, queued_ticket(n))
        gate = threading.Barrier(3, timeout=10)

        def fake_llm(ns, prepared):
            gate.wait()  # proves all three reviews really overlap
            workset.update_json(Path(ns.workset_file_override),
                                lambda d: d.__setitem__("seen", prepared.number))
            return decision(prepared.number)

        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            done = worker.drain(self.db, {"pr": self.ns}, parallel=3)
        self.assertEqual(done, 3)
        for n in (1, 2, 3):
            self.assertEqual(self.db.get("reviewed", "pr", n)["seen"], n)

    def test_a_slow_review_never_idles_the_other_slots(self) -> None:
        # a barrier round would wait for #1 before ever starting #3;
        # the top-up must run #3 (pushed mid-drain) while #1 still holds
        # its slot, or --parallel is parallel in name only
        import threading
        import time as _time
        release = threading.Event()

        def fake_llm(ns, prepared):
            if prepared.number == 1:
                self.assertTrue(release.wait(10), "top-up never happened")
            elif prepared.number == 2:
                self.db.push("queued", "pr", 3, queued_ticket(3))
            elif prepared.number == 3:
                release.set()
            return decision(prepared.number)

        for n in (1, 2):
            self.db.push("queued", "pr", n, queued_ticket(n))
        t0 = _time.monotonic()
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            done = worker.drain(self.db, {"pr": self.ns}, parallel=2)
        self.assertEqual(done, 3)
        self.assertLess(_time.monotonic() - t0, 5)

    def test_forced_tickets_are_claimed_first(self) -> None:
        order = []
        for n in (1, 2, 3):
            ticket = queued_ticket(n)
            ticket["forced"] = n == 3
            self.db.push("queued", "pr", n, ticket)

        def fake_llm(ns, prepared):
            order.append(prepared.number)
            return decision(prepared.number)

        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            worker.drain(self.db, {"pr": self.ns})
        self.assertEqual(order, [3, 1, 2])

    def test_broken_ticket_lands_in_error_and_does_not_starve(self) -> None:
        # A ticket the worker cannot even read must not return to
        # queued/: sorted first, it would be re-claimed on every pass
        # and the worker would never review anything again.
        self.db.push("queued", "pr", 1, {"title": "broken: no prepared"})
        self.db.push("queued", "pr", 2, queued_ticket(2))
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=lambda ns, p: decision(p.number)):
            done = worker.drain(self.db, {"pr": self.ns})
        self.assertEqual(done, 1)
        self.assertEqual(self.db.find("pr", 2), "reviewed")
        t = self.db.get("error", "pr", 1)
        self.assertIn("prepared", t["error"])
        self.assertTrue(t["llm_at"])


if __name__ == "__main__":
    unittest.main()
