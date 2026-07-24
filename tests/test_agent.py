"""agent: gate outcomes, backoff, limit and requests become filedb tickets."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent  # noqa: E402
import fairy  # noqa: E402
import filedb  # noqa: E402

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def make_pr(n: int) -> dict:
    return {"number": n, "title": f"t{n}", "user": {"login": "a"},
            "updated_at": "2026-07-19T10:00:00Z", "head": {"sha": f"h{n}"},
            "html_url": f"https://forge/pr/{n}"}


def prepared_for(pr: dict) -> fairy.PreparedPR:
    return fairy.PreparedPR(
        pr=pr, number=pr["number"], title=pr["title"], author="a",
        auto_merge="-", last_activity=None, base_reason="review",
        discussion=[], reviewer_username="fairy")


def gate_skip(pr: dict, reason: str = "no activity", **fields) -> fairy.Decision:
    return fairy.Decision(pr["number"], pr["title"], "a", "-", "skip",
                          reason, None, **fields)


class AgentCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))
        self.ns = fairy.parse_args(["--owner", "o", "--repo", "r"])
        self.prepare = mock.Mock(side_effect=lambda ns, pr, **kw: prepared_for(pr))

    def scan(self, prs: list[dict], forced: set[int] = frozenset()) -> None:
        with mock.patch.object(fairy, "list_open_prs", return_value=prs), \
                mock.patch.object(fairy, "safe_prepare_pr", self.prepare), \
                mock.patch.object(fairy, "get_self_login", return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, self.ns, None, now=NOW)
            if forced:  # requests are consumed by scan_pass itself
                pass

    def age(self, state: str, kind: str, number: int, hours: float) -> None:
        data = self.db.get(state, kind, number)
        data["state_changed_at"] = (NOW - timedelta(hours=hours)).isoformat()
        self.db._write(self.db._path(state, kind, number), data)


class TicketRoutingTests(AgentCase):
    def test_prepared_goes_to_queued_with_payload(self) -> None:
        self.scan([make_pr(1)])
        t = self.db.get("queued", "pr", 1)
        self.assertEqual(t["title"], "t1")
        self.assertEqual(t["skip_backoff_h"], 0)
        self.assertEqual(t["prepared"]["pr"]["head"]["sha"], "h1")
        prepared = fairy.prepared_pr_from_dict(t["prepared"])
        self.assertEqual(prepared.number, 1)

    def test_limit_caps_queued_and_persists_nothing(self) -> None:
        self.ns.limit = 2
        self.scan([make_pr(n) for n in (1, 2, 3)])
        self.assertEqual(self.db.list_state("queued"), [("pr", 1), ("pr", 2)])
        self.assertIsNone(self.db.find("pr", 3))

    def test_gate_outcomes_land_in_their_directories(self) -> None:
        prs = [make_pr(n) for n in (1, 2, 3, 4)]
        outcomes = {
            1: gate_skip(prs[0], "already approved", merge_ready=True),
            2: gate_skip(prs[1], "ci red", cancelled_ci_contexts=("job1",)),
            3: gate_skip(prs[2], "needs human", external_approvers=("dev",)),
            4: gate_skip(prs[3], "no activity"),
        }
        self.prepare.side_effect = lambda ns, pr, **kw: outcomes[pr["number"]]
        self.scan(prs)
        self.assertEqual(self.db.list_state("merge-ready"), [("pr", 1)])
        self.assertEqual(self.db.list_state("ci-blocked"), [("pr", 2)])
        self.assertEqual(self.db.list_state("awaiting-approver"), [("pr", 3)])
        self.assertEqual(self.db.list_state("skipped"), [("pr", 4)])
        self.assertEqual(self.db.get("ci-blocked", "pr", 2)
                         ["cancelled_ci_contexts"], ["job1"])

    def test_gate_skip_does_not_clobber_the_archive(self) -> None:
        self.db.push("posted", "pr", 1, {"posted": True})
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(pr)
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "posted")


class BackoffTests(AgentCase):
    def test_llm_skip_waits_out_its_backoff_untouched(self) -> None:
        self.db.push("skipped", "pr", 1, {"llm_at": NOW.isoformat(),
                                          "skip_backoff_h": 0})
        self.age("skipped", "pr", 1, hours=1)  # within the 24h minimum
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "skipped")

    def test_expired_backoff_requeues_with_doubled_wait(self) -> None:
        self.db.push("skipped", "pr", 1, {"llm_at": NOW.isoformat(),
                                          "skip_backoff_h": 24})
        self.age("skipped", "pr", 1, hours=49)  # past the 48h doubled wait
        self.scan([make_pr(1)])
        t = self.db.get("queued", "pr", 1)
        self.assertEqual(t["skip_backoff_h"], 48)
        self.assertIsNone(self.db.get("skipped", "pr", 1))


class ReuseTests(AgentCase):
    def test_standing_reviewed_verdict_suppresses_rereview(self) -> None:
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": "h1"})
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "reviewed")

    def test_stale_reviewed_verdict_is_requeued(self) -> None:
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "2026-07-01T00:00:00Z",  # PR changed since
            "expected_head_ref": "old"})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")


class LifecycleTests(AgentCase):
    def test_closed_item_is_cancelled(self) -> None:
        self.db.push("queued", "pr", 9, {"title": "gone"})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 9), "cancelled")
        self.assertEqual(self.db.get("cancelled", "pr", 9)["reason"], "not open")

    def test_request_forces_a_ticket_past_gates_and_limit(self) -> None:
        self.ns.limit = 1
        self.db.push("requests", "pr", 3, {"action": "rerun"})
        # gates would skip #3; the request must override them
        self.prepare.side_effect = lambda ns, pr, **kw: (
            prepared_for(pr) if pr["number"] in ns.force_review_prs
            or pr["number"] == 1 else gate_skip(pr))
        self.scan([make_pr(1), make_pr(2), make_pr(3)])
        self.assertEqual(self.db.find("pr", 3), "queued")
        self.assertTrue(self.db.get("queued", "pr", 3)["forced"])
        self.assertIsNone(self.db.get("requests", "pr", 3))
        self.assertEqual(self.ns.force_review_prs, set())  # undone after the pass

    def test_old_settled_tickets_are_pruned_open_ones_kept(self) -> None:
        self.db.push("posted", "pr", 1, {})   # still open -> kept
        self.db.push("posted", "pr", 99, {})  # closed + old -> pruned
        self.age("posted", "pr", 1, hours=24 * 30)
        self.age("posted", "pr", 99, hours=24 * 30)
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(pr)
        self.scan([make_pr(1)])
        self.assertIsNotNone(self.db.get("posted", "pr", 1))
        self.assertIsNone(self.db.get("posted", "pr", 99))


if __name__ == "__main__":
    unittest.main()
