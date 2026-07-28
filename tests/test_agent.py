"""agent: gate outcomes, backoff, limit and requests become filedb tickets."""

from __future__ import annotations

import sys
import tempfile
import unittest
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent  # noqa: E402
import fairy  # noqa: E402
import forge_gcli  # noqa: E402
import filedb  # noqa: E402
import issue_fairy  # noqa: E402

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def make_pr(n: int) -> dict:
    return {"number": n, "title": f"t{n}", "user": {"login": "a"},
            "updated_at": "2026-07-19T10:00:00Z",
            "head": {"sha": f"h{n}", "ref": f"b{n}"},
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

    def scan(self, prs: list[dict], fetch=None) -> None:
        with mock.patch.object(fairy, "list_open_prs", return_value=prs), \
                mock.patch.object(fairy, "get_pr",
                                  side_effect=fetch or (lambda ns, n: make_pr(n))), \
                mock.patch.object(fairy, "safe_prepare_pr", self.prepare), \
                mock.patch.object(forge_gcli, "self_login", return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, self.ns, None, now=NOW)

    def age(self, state: str, kind: str, number: int, hours: float) -> None:
        data = self.db.get(state, kind, number)
        data["state_changed_at"] = (NOW - timedelta(hours=hours)).isoformat()
        for field in ("llm_at", "snoozed_at"):
            if data.get(field):  # the backoff window is measured from these
                data[field] = data["state_changed_at"]
        self.db._write(self.db.path(state, kind, number), data)


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

    def test_forge_forced_review_bypasses_the_limit(self) -> None:
        """A REQUEST_REVIEW / mention must not starve behind stale
        eligible items filling --limit (production: FFmpeg #23896)."""
        self.ns.limit = 1
        self.prepare.side_effect = lambda ns, pr, **kw: (
            dataclasses.replace(prepared_for(pr), forced_review=True)
            if pr["number"] == 2 else prepared_for(pr))
        self.scan([make_pr(1), make_pr(2)])
        self.assertEqual(self.db.find("pr", 1), "queued")
        self.assertEqual(self.db.find("pr", 2), "queued")

    def test_gate_outcomes_land_in_their_directories(self) -> None:
        prs = [make_pr(n) for n in (1, 2, 3, 4)]
        outcomes = {
            1: gate_skip(prs[0], "already approved", merge_ready=True,
                         approved_at=NOW - timedelta(days=12)),
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
        # attention rows are acted on in the forge web UI: the link
        # must ride on the ticket
        self.assertEqual(self.db.get("merge-ready", "pr", 1)["html_url"],
                         "https://forge/pr/1")
        self.assertEqual(self.db.get("merge-ready", "pr", 1)["approved_at"],
                         (NOW - timedelta(days=12)).isoformat())

    def test_pr_prepare_failure_is_a_paced_error_not_a_skip(self) -> None:
        # like the issue side: a row, a summary line, ERROR_RETRY_H --
        # and never a clobbered standing verdict
        self.prepare.side_effect = lambda ns, pr, **kw: fairy.Decision(
            pr["number"], pr["title"], "a", "-", "error",
            "gcli timeout", None, "error", "")
        self.scan([make_pr(1)])
        self.assertIn("gcli timeout", self.db.get("error", "pr", 1)["error"])
        self.prepare.reset_mock()
        self.scan([make_pr(1)])  # within ERROR_RETRY_H: no refetch churn
        self.prepare.assert_not_called()
        self.db.push("reviewed", "pr", 2, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "old", "expected_head_ref": "old"})
        self.prepare.side_effect = lambda ns, pr, **kw: fairy.Decision(
            pr["number"], pr["title"], "a", "-", "error",
            "forge 500", None, "error", "")
        self.scan([make_pr(2)])
        self.assertEqual(self.db.find("pr", 2), "reviewed")  # verdict kept

    def test_unchanged_gate_outcome_is_not_rewritten(self) -> None:
        # a rewrite per scan would defeat the TUI's dir-mtime gate and
        # re-stamp state_changed_at every pass
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(
            pr, "ci red", cancelled_ci_contexts=("job1",))
        self.scan([make_pr(1)])
        stamp = self.db.path("ci-blocked", "pr", 1).stat().st_mtime_ns
        self.scan([make_pr(1)])
        self.assertEqual(
            self.db.path("ci-blocked", "pr", 1).stat().st_mtime_ns, stamp)
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(
            pr, "ci red", cancelled_ci_contexts=("job1", "job2"))
        self.scan([make_pr(1)])  # outcome changed: must be rewritten
        self.assertEqual(self.db.get("ci-blocked", "pr", 1)
                         ["cancelled_ci_contexts"], ["job1", "job2"])

    def test_gate_skip_does_not_clobber_the_archive(self) -> None:
        self.db.push("posted", "pr", 1, {"posted": True})
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(pr)
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "posted")


def llm_skip(backoff: float, updated: str = "2026-07-19T10:00:00Z",
             head: str = "h1", last_iso: str | None = None) -> dict:
    return {"llm_at": NOW.isoformat(), "skip_backoff_h": backoff,
            "expected_updated_at": updated, "expected_head_ref": head,
            "last_activity_iso": last_iso}


class BackoffTests(AgentCase):
    def test_plain_operator_skip_is_reconsidered_next_scan(self) -> None:
        self.db.push("skipped", "pr", 1,
                     {"reason": "operator skip", "skip_backoff_h": 24})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")
        self.assertEqual(self.db.get("queued", "pr", 1)["skip_backoff_h"], 24)

    def test_llm_skip_waits_out_its_backoff_untouched(self) -> None:
        self.db.push("skipped", "pr", 1, llm_skip(0))
        self.age("skipped", "pr", 1, hours=1)  # within the 24h minimum
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "skipped")

    def test_expired_backoff_requeues_with_doubled_wait(self) -> None:
        self.db.push("skipped", "pr", 1, llm_skip(24))
        self.age("skipped", "pr", 1, hours=49)  # past the 48h doubled wait
        self.scan([make_pr(1)])
        t = self.db.get("queued", "pr", 1)
        self.assertEqual(t["skip_backoff_h"], 48)
        self.assertIsNone(self.db.get("skipped", "pr", 1))

    def test_new_activity_bypasses_the_window_without_doubling(self) -> None:
        # A new comment must reach the gates now, not after the 48h
        # window; the backoff only doubles for waits actually served.
        self.db.push("skipped", "pr", 1,
                     llm_skip(24, updated="old", last_iso="2026-07-01T00:00:00+00:00"))
        self.age("skipped", "pr", 1, hours=1)
        self.prepare.side_effect = lambda ns, pr, **kw: dataclasses.replace(
            prepared_for(pr), last_activity=datetime(2026, 7, 19, tzinfo=timezone.utc))
        self.scan([make_pr(1)])
        self.assertEqual(self.db.get("queued", "pr", 1)["skip_backoff_h"], 24)

    def test_new_head_bypasses_the_window_too(self) -> None:
        self.db.push("skipped", "pr", 1, llm_skip(24, updated="old",
                                                  head="old-sha"))
        self.age("skipped", "pr", 1, hours=1)
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_operator_skip_snoozes_with_doubling(self) -> None:
        # Approved cutover divergence D5: s means "not now", not "not
        # this run" -- a >=24h doubling wait that any real activity
        # bypasses, instead of the old re-review-every-run.
        self.db.push("skipped", "pr", 1, {
            "llm_at": NOW.isoformat(), "reason": "operator skip",
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": "h1"})
        self.age("skipped", "pr", 1, hours=1)
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()  # snoozing
        self.age("skipped", "pr", 1, hours=25)
        self.scan([make_pr(1)])
        self.assertEqual(self.db.get("queued", "pr", 1)["skip_backoff_h"], 24)

    def test_s_on_a_queued_row_snoozes_without_an_llm_run(self) -> None:
        # s before the worker got there: no llm_at exists, only the
        # press; the item must not bounce straight back to queued
        self.db.push("skipped", "pr", 1, {
            "snoozed_at": NOW.isoformat(), "reason": "operator skip",
            "skip_backoff_h": 0,
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": "h1"})
        self.age("skipped", "pr", 1, hours=1)
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "skipped")
        self.age("skipped", "pr", 1, hours=25)
        self.scan([make_pr(1)])  # snooze served: normal life resumes
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_label_edit_does_not_bypass_the_window(self) -> None:
        # A label/milestone edit bumps updated_at but neither head nor
        # discussion: the wait must hold (old compute_llm_skip_backoff
        # keyed on exactly this), and the refreshed updated_at makes the
        # next scans cheap again.
        self.db.push("skipped", "pr", 1, llm_skip(24, updated="old"))
        self.age("skipped", "pr", 1, hours=1)
        self.scan([make_pr(1)])  # prepared.last_activity None == stored None
        self.assertEqual(self.db.find("pr", 1), "skipped")
        self.assertEqual(self.db.get("skipped", "pr", 1)["expected_updated_at"],
                         "2026-07-19T10:00:00Z")

    def test_operator_skip_of_an_old_verdict_still_snoozes(self) -> None:
        # llm_at may be days old when the operator presses s; the snooze
        # is measured from the press, not the review
        t = {"llm_at": (NOW - timedelta(hours=100)).isoformat(),
             "snoozed_at": (NOW - timedelta(hours=1)).isoformat(),
             "reason": "operator skip",
             "review": {"classification": "moderate_issues"},
             "expected_updated_at": "2026-07-19T10:00:00Z",
             "expected_head_ref": "h1"}
        self.db.push("skipped", "pr", 1, t)
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "skipped")

    def test_x_on_a_queued_item_sticks(self) -> None:
        self.scan([make_pr(1)])  # queued ticket now carries the guard
        self.assertTrue(self.db.try_move(
            "queued", "cancelled", "pr", 1,
            mutate=lambda d: d.update(reason="operator cancel")))
        self.prepare.reset_mock()
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "cancelled")

    def test_label_refresh_does_not_extend_the_window(self) -> None:
        # The wait is measured from llm_at: however often labels get
        # edited (each refresh re-stamps state_changed_at), the item
        # re-enters the gates once the original window has passed.
        self.db.push("skipped", "pr", 1, llm_skip(0, updated="old"))
        data = self.db.get("skipped", "pr", 1)
        data["llm_at"] = (NOW - timedelta(hours=25)).isoformat()
        self.db._write(self.db.path("skipped", "pr", 1), data)  # freshly refreshed, old LLM run
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")


class ReuseTests(AgentCase):
    def test_standing_reviewed_verdict_suppresses_rereview(self) -> None:
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": "h1"})
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "reviewed")

    def test_standing_skip_with_labels_verdict_is_not_rereviewed(self) -> None:
        # skip+labels sits in reviewed/ awaiting the operator; re-queueing
        # it would burn one LLM run per scan pass, forever.
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "skip",
                       "label_changes": [{"label": "needs docs", "op": "add"}]},
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": "h1"})
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "reviewed")

    def test_closure_cancel_from_a_short_listing_revives(self) -> None:
        # a transient pagination glitch cancels with reason "not open";
        # when the item is listed again the verdict work must resume --
        # only operator cancels stick
        self.db.push("cancelled", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "reason": "not open",
            "expected_updated_at": "2026-07-19T10:00:00Z"})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_operator_cancel_sticks_until_new_activity(self) -> None:
        self.db.push("cancelled", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "reason": "operator cancel",
            "expected_updated_at": "2026-07-19T10:00:00Z"})
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "cancelled")
        self.db.try_move("cancelled", "cancelled", "pr", 1,
                         mutate=lambda d: d.update(
                             expected_updated_at="2026-07-01T00:00:00Z"))
        self.scan([make_pr(1)])  # the PR changed since the cancel
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_gate_ticket_records_the_change_guard(self) -> None:
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(
            pr, "ci red", cancelled_ci_contexts=("job1",))
        self.scan([make_pr(1)])
        ticket = self.db.get("ci-blocked", "pr", 1)
        self.assertEqual(ticket["expected_updated_at"], "2026-07-19T10:00:00Z")
        self.assertEqual(ticket["head_branch"], "b1")

    def test_queued_ticket_carries_author_and_head_branch(self) -> None:
        self.scan([make_pr(1)])
        ticket = self.db.get("queued", "pr", 1)
        self.assertEqual(ticket["author"], "a")
        self.assertEqual(ticket["head_branch"], "b1")

    def test_stale_reviewed_verdict_is_requeued(self) -> None:
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "2026-07-01T00:00:00Z",  # PR changed since
            "expected_head_ref": "old"})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_s_and_x_during_the_slow_prepare_are_not_clobbered(self) -> None:
        # Same race as the y test below, for the decline actions: the
        # operator said no during the prepare; the fresh queued ticket
        # must not resurrect the item.
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "old", "expected_head_ref": "old"})

        def prepare_and_skip(ns, pr, **kw):
            self.db.try_move("reviewed", "skipped", "pr", pr["number"],
                             mutate=lambda d: d.update(reason="operator skip"))
            return prepared_for(pr)

        self.prepare.side_effect = prepare_and_skip
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "skipped")

    def test_y_press_during_the_slow_prepare_is_not_clobbered(self) -> None:
        # The IN_FLIGHT check runs before prepare; prepare takes seconds.
        # An operator y (reviewed -> outgoing) in that window must not be
        # popped by the requeue routing: the pending send would vanish.
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "old", "expected_head_ref": "old"})

        def prepare_and_y(ns, pr, **kw):
            self.db.try_move("reviewed", "outgoing", "pr", pr["number"])
            return prepared_for(pr)

        self.prepare.side_effect = prepare_and_y
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "outgoing")


class LifecycleTests(AgentCase):
    def test_closed_item_is_cancelled(self) -> None:
        self.db.push("queued", "pr", 9, {"title": "gone"})
        self.db.push("merge-ready", "pr", 8, {"title": "merged meanwhile"})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 9), "cancelled")
        self.assertEqual(self.db.get("cancelled", "pr", 9)["reason"], "not open")
        self.assertEqual(self.db.find("pr", 8), "cancelled")  # attention too

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

    def test_request_for_an_unlisted_item_is_fetched_and_consumed(self) -> None:
        # A closed/merged item never appears in the open listing; the
        # request must fetch it directly or it wedges requests/ forever.
        self.db.push("requests", "pr", 9, {"action": "rerun"})
        self.prepare.side_effect = lambda ns, pr, **kw: (
            gate_skip(pr, "not open") if pr["number"] == 9 else prepared_for(pr))
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 9), "skipped")  # gate outcome shown
        self.assertIsNone(self.db.get("requests", "pr", 9))

    def test_failed_forced_fetch_becomes_an_error_ticket(self) -> None:
        self.db.push("requests", "pr", 9, {"action": "rerun"})

        def fetch(ns, n):
            raise RuntimeError("404")

        self.scan([make_pr(1)], fetch=fetch)
        self.assertIn("404", self.db.get("error", "pr", 9)["error"])
        self.assertIsNone(self.db.get("requests", "pr", 9))

    def test_prune_leaves_the_other_kinds_tickets_alone(self) -> None:
        # split pr-only/issue-only agents share one db root: the pr
        # agent's keep-set knows nothing about open issues
        self.db.push("skipped", "issue", 9, {"llm_at": "x"})
        self.age("skipped", "issue", 9, hours=24 * 30)
        self.scan([make_pr(1)])
        self.assertIsNotNone(self.db.get("skipped", "issue", 9))

    def test_torn_request_is_consumed_not_wedging(self) -> None:
        self.db.push("requests", "pr", 9, {"action": "rerun"})
        self.db.path("requests", "pr", 9).write_text("{ torn")
        self.scan([make_pr(1)])  # must not raise
        self.assertIsNone(self.db.get("requests", "pr", 9))

    def test_request_for_an_unscanned_kind_is_left_alone(self) -> None:
        # a pr-only agent shares the db with issue tickets: an issue
        # rerun request is another agent's to satisfy, not ours to eat
        self.db.push("posted", "issue", 9, {"title": "old"})
        self.db.push("requests", "issue", 9, {"action": "rerun"})
        self.scan([make_pr(1)])
        self.assertEqual(self.db.get("requests", "issue", 9)["action"],
                         "rerun")

    def test_sample_requests_create_parallel_evaluations(self) -> None:
        # 3 samples = 3 queued tickets from one prepare; no base
        # evaluation nobody asked for
        for token in ("7s1", "7s2", "7s3"):
            self.db.push("requests", "pr", token, {"action": "rerun"})
        self.scan([make_pr(7)])
        self.assertEqual(self.db.list_state("queued"),
                         [("pr", "7s1"), ("pr", "7s2"), ("pr", "7s3")])
        for token in ("7s1", "7s2", "7s3"):
            t = self.db.get("queued", "pr", token)
            self.assertEqual(fairy.prepared_pr_from_dict(
                t["prepared"]).number, 7)
            self.assertIsNone(self.db.get("requests", "pr", token))

    def test_skipped_sample_never_requeues(self) -> None:
        # a 3-sample evaluation must not loop into 3 more evaluations
        self.db.push("skipped", "pr", "7s1",
                     dict(llm_skip(0), reason="operator skip"))
        self.age("skipped", "pr", "7s1", hours=100)  # any wait long served
        self.scan([make_pr(7)])
        self.assertEqual(self.db.find("pr", "7s1"), "skipped")

    def test_sample_of_a_gate_skipped_item_becomes_an_error_ticket(self) -> None:
        self.db.push("requests", "pr", "7s1", {"action": "rerun"})
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(pr, "not open")
        self.scan([make_pr(7)])
        self.assertIn("gate-skipped", self.db.get("error", "pr", "7s1")["error"])
        self.assertIsNone(self.db.get("requests", "pr", "7s1"))

    def test_old_settled_tickets_are_pruned_open_ones_kept(self) -> None:
        self.db.push("posted", "pr", 1, {})   # still open -> kept
        self.db.push("posted", "pr", 99, {})  # closed + old -> pruned
        self.age("posted", "pr", 1, hours=24 * 30)
        self.age("posted", "pr", 99, hours=24 * 30)
        self.prepare.side_effect = lambda ns, pr, **kw: gate_skip(pr)
        self.scan([make_pr(1)])
        self.assertIsNotNone(self.db.get("posted", "pr", 1))
        self.assertIsNone(self.db.get("posted", "pr", 99))

    def test_each_side_prunes_by_its_own_retention(self) -> None:
        self.ns.workset_retention_days = 30.0
        issue_ns = issue_fairy.parse_args(
            ["--owner", "o", "--repo", "r", "--workset-retention-days", "5"])
        for kind in ("pr", "issue"):
            self.db.push("posted", kind, 9, {})
            self.age("posted", kind, 9, hours=24 * 10)
        with mock.patch.object(fairy, "list_open_prs", return_value=[]), \
                mock.patch.object(issue_fairy, "list_open_issues",
                                  return_value=[]), \
                mock.patch.object(forge_gcli, "self_login",
                                  return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, self.ns, issue_ns, now=NOW)
        self.assertIsNotNone(self.db.get("posted", "pr", 9))
        self.assertIsNone(self.db.get("posted", "issue", 9))


def verdict_ticket(n: int, classification: str = "moderate_issues",
                   msg: str = "m", labels: list | None = None,
                   **fields) -> dict:
    t = {"title": f"t{n}", "author": "a", "skip_backoff_h": 24,
         "review": {"classification": classification, "message": msg,
                    "label_changes": labels or []},
         "expected_updated_at": "2026-07-19T10:00:00Z",
         "expected_head_ref": f"h{n}", "llm_at": NOW.isoformat()}
    t.update(fields)
    return t


class TicketDecisionTests(AgentCase):
    def test_auto_merge_rides_the_ticket_into_the_y_prompt(self) -> None:
        d = agent.ticket_decision("pr", 1, dict(
            verdict_ticket(1, "approve"), auto_merge="merge"))
        self.assertEqual(d.auto_merge, "merge")
        self.assertEqual(d.action, "approve")
        self.assertIn("auto-merge scheduled: approving MERGES the PR",
                      fairy.manual_action_description(d))
        plain = agent.ticket_decision("pr", 2, verdict_ticket(2, "approve"))
        self.assertNotIn("MERGES", fairy.manual_action_description(plain))

    def test_hand_edited_label_junk_cannot_raise(self) -> None:
        d = agent.ticket_decision("pr", 1, {
            "title": "t", "review": {
                "classification": "moderate_issues", "message": "m",
                "label_changes": [{"label": "ok", "op": "add"},
                                  {"lable": "typo key"}, "not a dict"]}})
        self.assertEqual([c.label for c in d.label_changes], ["ok", ""])


class SendCase(AgentCase):
    def send(self, *, pr_ns=None, issue_ns=None, dry_run=False) -> None:
        with mock.patch.object(agent.gcli_cache, "load_cache",
                               return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.send_pass(self.db, pr_ns, issue_ns, dry_run=dry_run)


class SendTests(SendCase):
    def test_actionable_outgoing_is_posted(self) -> None:
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value=None), \
                mock.patch.object(fairy, "submit_decision_action",
                                  return_value=True) as submit:
            self.send(pr_ns=self.ns)
        decision = submit.call_args.args[2]
        self.assertEqual(decision.action, "comment")
        # the ticket's guard rides on the rebuilt decision
        self.assertEqual(decision.expected_pr_updated_at, "2026-07-19T10:00:00Z")
        self.assertEqual(decision.expected_head_ref, "h1")
        self.assertTrue(self.db.get("posted", "pr", 1)["posted_at"])
        self.assertIsNone(self.db.get("outgoing", "pr", 1))

    def test_guard_failure_manual_returns_to_reviewed_with_note(self) -> None:
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value="PR updated_at changed"), \
                mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns)
        submit.assert_not_called()
        t = self.db.get("reviewed", "pr", 1)
        self.assertEqual(t["send_blocked"], "PR updated_at changed")

    def test_guard_failure_auto_mode_skips_without_stalling(self) -> None:
        self.ns.approve = True
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value="PR head changed"), \
                mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns)
        submit.assert_not_called()
        t = self.db.get("skipped", "pr", 1)
        self.assertEqual(t["send_blocked"], "PR head changed")
        self.assertNotIn("llm_at", t)  # next scan re-gates it immediately
        self.assertEqual(t["skip_backoff_h"], 24)  # earned history kept

    def test_approve_promotes_only_actionable_reviewed_verdicts(self) -> None:
        self.ns.approve = True
        self.db.push("reviewed", "pr", 1, verdict_ticket(1))
        self.db.push("reviewed", "pr", 2, verdict_ticket(2, "skip"))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value=None), \
                mock.patch.object(fairy, "submit_decision_action",
                                  return_value=True):
            self.send(pr_ns=self.ns)
        self.assertEqual(self.db.find("pr", 1), "posted")
        self.assertEqual(self.db.find("pr", 2), "reviewed")

    def test_unpostable_outgoing_goes_back_to_reviewed(self) -> None:
        self.db.push("outgoing", "pr", 1, verdict_ticket(1, "skip"))
        with mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns)
        submit.assert_not_called()
        t = self.db.get("reviewed", "pr", 1)
        self.assertEqual(t["send_blocked"], "nothing to post")

    def test_reviewed_crash_remnant_is_not_repromoted(self) -> None:
        # crash between finish's dst-write and src-unlink: the item is
        # in outgoing AND reviewed; only the later state is real
        self.ns.approve = True
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        self.db.path("reviewed", "pr", 1).write_text(
            self.db.path("outgoing", "pr", 1).read_text())
        agent.promote_reviewed(self.db, "pr")
        self.assertIsNotNone(self.db.get("outgoing", "pr", 1))
        self.assertIsNotNone(self.db.get("reviewed", "pr", 1))  # the scan pass reaps it, not promote

    def test_dry_run_never_promotes_reviewed_to_outgoing(self) -> None:
        # promotion is persistent: a dry preview must not stage posts
        # that a later normal run would then send without consent
        self.ns.approve = True
        self.db.push("reviewed", "pr", 1, verdict_ticket(1))
        with mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns, dry_run=True)
        submit.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "reviewed")

    def test_dry_run_posts_nothing_and_keeps_outgoing(self) -> None:
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        with mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns, dry_run=True)
        submit.assert_not_called()
        self.assertEqual(self.db.find("pr", 1), "outgoing")

    def test_dry_run_send_fires_no_event_the_agent_would_wake_on(self) -> None:
        import os as _os
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        d = self.db.root / "outgoing"
        _os.utime(d, (100.0, 100.0))
        stamp = d.stat().st_mtime_ns
        self.send(pr_ns=self.ns, dry_run=True)
        self.assertEqual(d.stat().st_mtime_ns, stamp)  # else: wake loop

    def test_deleted_outgoing_file_is_never_posted(self) -> None:
        # the operator deleting the file IS the veto: an unclaimable
        # ticket cannot be posted, however stale the send pass's listing
        self.db.push("outgoing", "pr", 1, verdict_ticket(1))
        self.db.path("outgoing", "pr", 1).unlink()
        with mock.patch.object(filedb.Db, "list_state",
                               return_value=[("pr", 1)]), \
                mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns)
        submit.assert_not_called()

    def test_operator_edit_is_what_gets_posted(self) -> None:
        self.db.push("outgoing", "pr", 1, verdict_ticket(1, msg="original"))
        self.db.try_move("outgoing", "outgoing", "pr", 1,
                         mutate=lambda d: d["review"].update(
                             message="edited by hand"))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value=None), \
                mock.patch.object(fairy, "submit_decision_action",
                                  return_value=True) as submit:
            self.send(pr_ns=self.ns)
        self.assertEqual(submit.call_args.args[2].llm_message, "edited by hand")

    def test_label_only_verdict_posts_labels_then_lands_in_posted(self) -> None:
        labels = [{"label": "needs docs", "op": "add", "reason": "", "post": False}]
        self.db.push("outgoing", "pr", 1, verdict_ticket(1, "skip", labels=labels))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value=None), \
                mock.patch.object(fairy, "apply_triage_labels",
                                  return_value=True) as apply, \
                mock.patch.object(fairy, "submit_decision_action") as submit:
            self.send(pr_ns=self.ns)
        submit.assert_not_called()  # nothing to comment/approve
        self.assertFalse(apply.call_args.kwargs["skip_guard"])
        self.assertEqual(self.db.find("pr", 1), "posted")

    def test_label_only_guard_failure_posts_nothing(self) -> None:
        labels = [{"label": "needs docs", "op": "add", "reason": "", "post": False}]
        self.db.push("outgoing", "pr", 1, verdict_ticket(1, "skip", labels=labels))
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value=None), \
                mock.patch.object(fairy, "apply_triage_labels",
                                  return_value=False):
            self.send(pr_ns=self.ns)
        t = self.db.get("reviewed", "pr", 1)
        self.assertIn("changed during submit", t["send_blocked"])

    def test_issue_outgoing_posts_through_the_issue_seam(self) -> None:
        issue_ns = issue_fairy.parse_args(["--owner", "o", "--repo", "r"])
        self.db.push("outgoing", "issue", 5, verdict_ticket(
            5, "reply", expected_head_ref=None))
        with mock.patch.object(issue_fairy, "check_issue_still_unchanged",
                               return_value=None), \
                mock.patch.object(issue_fairy, "submit_issue_decision",
                                  return_value=True) as submit:
            self.send(issue_ns=issue_ns)
        decision = submit.call_args.args[1]
        self.assertEqual(decision.action, "comment")
        self.assertEqual(self.db.find("issue", 5), "posted")


class ForcedOnlyTests(AgentCase):
    def test_forced_only_never_cancels_or_prunes_other_tickets(self) -> None:
        # A forced-only run's candidate list is not the open listing:
        # every other item would look closed and lose its ticket.
        self.ns.forced_only = True
        self.ns.force_review_prs = {7}
        self.db.push("reviewed", "pr", 1, {"review": {"classification": "reply"}})
        self.db.push("queued", "pr", 2, {"title": "t"})
        self.db.push("skipped", "pr", 3, {"llm_at": "x", "skip_backoff_h": 24})
        self.age("skipped", "pr", 3, hours=24 * 30)  # over retention age
        with mock.patch.object(fairy, "get_pr",
                               side_effect=lambda ns, n: make_pr(n)), \
                mock.patch.object(fairy, "safe_prepare_pr", self.prepare), \
                mock.patch.object(forge_gcli, "self_login", return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, self.ns, None, now=NOW)
        self.assertEqual(self.db.find("pr", 1), "reviewed")
        self.assertEqual(self.db.find("pr", 2), "queued")
        self.assertEqual(self.db.find("pr", 3), "skipped")  # backoff memory kept

    def test_forced_only_fetches_named_items_without_listing(self) -> None:
        self.ns.forced_only = True
        self.ns.force_review_prs = {7}
        with mock.patch.object(fairy, "list_open_prs") as listing, \
                mock.patch.object(fairy, "get_pr",
                                  side_effect=lambda ns, n: make_pr(n)), \
                mock.patch.object(fairy, "safe_prepare_pr", self.prepare), \
                mock.patch.object(forge_gcli, "self_login", return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, self.ns, None, now=NOW)
        listing.assert_not_called()
        self.assertEqual(self.db.find("pr", 7), "queued")


class BackoffMathTests(unittest.TestCase):
    def test_floor_and_doubling(self) -> None:  # cutover audit G8
        self.assertEqual(agent.backoff_wait_h(0), 24.0)
        self.assertEqual(agent.backoff_wait_h(None), 24.0)
        self.assertEqual(agent.backoff_wait_h(24), 48.0)
        self.assertEqual(agent.backoff_wait_h(48), 96.0)


class PortedGateContractTests(AgentCase):
    """Ports from the deleted pipeline tests (cutover audit G2-G15)."""

    def test_limit_zero_queues_every_pr(self) -> None:
        self.ns.limit = 0
        self.scan([make_pr(n) for n in range(1, 6)])
        self.assertEqual(len(self.db.list_state("queued")), 5)

    def test_gate_outcomes_never_consume_limit_slots(self) -> None:
        # the --limit invariant: it caps LLM entries, so items the gates
        # turn away must not eat slots
        self.ns.limit = 2
        self.prepare.side_effect = lambda ns, pr, **kw: (
            gate_skip(pr) if pr["number"] % 2 else prepared_for(pr))
        self.scan([make_pr(n) for n in (1, 2, 3, 4, 5, 6)])
        self.assertEqual(self.db.list_state("queued"), [("pr", 2), ("pr", 4)])
        self.assertEqual(len(self.db.list_state("skipped")), 3)
        self.assertIsNone(self.db.find("pr", 6))  # over limit: nothing written

    def test_forced_number_reruns_despite_standing_verdict(self) -> None:
        self.db.push("reviewed", "pr", 1, {
            "review": {"classification": "moderate_issues"},
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": "h1"})
        self.ns.force_review_prs = {1}
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_gate_skip_ticket_never_serves_a_backoff_wait(self) -> None:
        self.db.push("skipped", "pr", 1, {"reason": "no activity"})  # no llm_at
        self.scan([make_pr(1)])
        self.prepare.assert_called_once()
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_forced_number_bypasses_a_served_backoff_window(self) -> None:
        self.db.push("skipped", "pr", 1, llm_skip(24))
        self.age("skipped", "pr", 1, hours=1)
        self.ns.force_review_prs = {1}
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_error_retry_is_paced_not_forgotten(self) -> None:
        self.db.push("error", "pr", 1, {"error": "boom"})
        self.age("error", "pr", 1, hours=1)
        self.scan([make_pr(1)])
        self.prepare.assert_not_called()  # within ERROR_RETRY_H: no spend
        self.assertEqual(self.db.find("pr", 1), "error")
        self.age("error", "pr", 1, hours=25)
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")  # and never forgotten

    def test_corrupt_timestamps_fail_open(self) -> None:
        self.db.push("skipped", "pr", 1, dict(llm_skip(24), llm_at="garbage"))
        self.scan([make_pr(1)])
        self.assertEqual(self.db.find("pr", 1), "queued")

    def test_naive_timestamps_never_crash_the_scan(self) -> None:
        # a hand-edited timestamp without timezone must not TypeError
        self.db.push("skipped", "pr", 1,
                     dict(llm_skip(24), llm_at="2026-01-01T00:00:00"))
        self.scan([make_pr(1)])  # months old: window served long ago
        self.assertEqual(self.db.find("pr", 1), "queued")


def make_issue(n: int) -> dict:
    return {"number": n, "title": f"i{n}",
            "updated_at": "2026-07-19T10:00:00Z", "html_url": f"u{n}"}


class IssueSideScanTests(AgentCase):
    """The kind-specific seams of the shared scan (cutover audit
    G13/G14): issues have no head guard and their own prepare."""

    def scan_issues(self, issues: list[dict], prepare) -> None:
        ins = issue_fairy.parse_args(["--owner", "o", "--repo", "r"])
        ins.limit = 1
        with mock.patch.object(issue_fairy, "list_open_issues",
                               return_value=issues), \
                mock.patch.object(issue_fairy, "prepare_issue",
                                  side_effect=prepare) as self.prepare_issue, \
                mock.patch.object(forge_gcli, "self_login", return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, None, ins, now=NOW)

    def test_issue_scan_gates_limit_and_ticket_shape(self) -> None:
        def prepare(ns, issue, **kw):
            if issue["number"] == 1:
                return fairy.Decision(1, "i1", "a", "-", "skip", "too new", None)
            return issue_fairy.PreparedIssue(
                issue=issue, number=issue["number"], title=issue["title"],
                author="a", last_activity=None, base_reason="stale",
                discussion=[], reviewer_username="fairy")

        self.scan_issues([make_issue(n) for n in (1, 2, 3)], prepare)
        self.assertEqual(self.db.list_state("queued"), [("issue", 2)])
        self.assertEqual(self.db.list_state("skipped"), [("issue", 1)])
        self.assertIsNone(self.db.find("issue", 3))  # over --limit 1
        prepared = issue_fairy.prepared_issue_from_dict(
            self.db.get("queued", "issue", 2)["prepared"])
        self.assertEqual(prepared.number, 2)

    def test_issue_standing_verdict_reused_without_head_guard(self) -> None:
        self.db.push("reviewed", "issue", 1, {
            "review": {"classification": "reply"},
            "expected_updated_at": "2026-07-19T10:00:00Z",
            "expected_head_ref": None})
        self.scan_issues([make_issue(1)], prepare=AssertionError)
        self.prepare_issue.assert_not_called()
        self.assertEqual(self.db.find("issue", 1), "reviewed")


class LogSummaryTests(AgentCase):
    def test_summary_names_appliable_verdicts_and_ci_details(self) -> None:
        self.db.push("reviewed", "pr", 5, dict(verdict_ticket(5),
                                               action="approve", title="t5"))
        self.db.push("merge-ready", "pr", 6, {"title": "t6"})
        self.db.push("ci-blocked", "pr", 7, {
            "title": "t7", "cancelled_ci_contexts": ["job1"],
            "blocked_ci_contexts": ["job2"]})
        with self.assertLogs(agent.logger, level="INFO") as logs:
            agent.log_summary(self.db)
        text = "\n".join(logs.output)
        self.assertIn("awaiting you: 1", text)
        self.assertIn("approve", text)
        self.assertIn("approved, ready to apply", text)
        self.assertIn("t6", text)
        self.assertIn("job1, job2", text)


class AskPassTests(AgentCase):
    """--ask: the pre-TUI prompt flow over actionable reviewed/ verdicts."""

    def ask(self, answers: list[str]) -> None:
        with mock.patch("builtins.input", side_effect=answers):
            agent.ask_pass(self.db, {"pr"})

    def test_answers_route_the_verdicts(self) -> None:
        for n in (1, 2, 3, 4, 6):
            self.db.push("reviewed", "pr", n, verdict_ticket(n))
        self.db.push("reviewed", "pr", 5, verdict_ticket(5, "skip"))  # unpostable: never asked
        self.ask(["y", "s", "l", "x", "S"])
        self.assertEqual(self.db.find("pr", 1), "outgoing")
        skipped = self.db.get("skipped", "pr", 2)
        self.assertEqual(skipped["reason"], "operator skip")
        self.assertNotIn("llm_at", skipped)  # one-shot: next scan reconsiders
        self.assertNotIn("snoozed_at", skipped)
        self.assertEqual(self.db.find("pr", 3), "reviewed")  # later
        self.assertEqual(self.db.find("pr", 4), "cancelled")
        self.assertEqual(self.db.find("pr", 5), "reviewed")
        snoozed = self.db.get("skipped", "pr", 6)
        self.assertEqual(snoozed["reason"], "operator snooze")
        self.assertTrue(snoozed["snoozed_at"])  # S: the >=24h doubling wait

    def test_quit_and_eof_stop_asking(self) -> None:
        for n in (1, 2):
            self.db.push("reviewed", "pr", n, verdict_ticket(n))
        self.ask(["q"])
        self.assertEqual(self.db.find("pr", 2), "reviewed")  # never asked
        self.ask([EOFError, "y"])  # EOF (^D) stops like q
        self.assertEqual(self.db.find("pr", 1), "reviewed")

    def test_garbage_answer_reprompts(self) -> None:
        self.db.push("reviewed", "pr", 1, verdict_ticket(1))
        self.ask(["bogus", "y"])
        self.assertEqual(self.db.find("pr", 1), "outgoing")

    def test_retry_reviews_again_and_reasks(self) -> None:
        self.db.push("reviewed", "pr", 1, verdict_ticket(1))

        def fresh_review():
            self.assertEqual(self.db.get("requests", "pr", 1)["action"],
                             "rerun")  # the request precedes the cycle
            self.db.push("reviewed", "pr", 1, verdict_ticket(1, title="fresh"))

        with mock.patch("builtins.input", side_effect=["r", "y"]):
            agent.ask_pass(self.db, {"pr"}, retry=fresh_review)
        self.assertEqual(self.db.get("outgoing", "pr", 1)["title"], "fresh")

    def test_retry_without_inline_worker_files_the_request(self) -> None:
        self.db.push("reviewed", "pr", 1, verdict_ticket(1))
        self.ask(["r"])
        self.assertEqual(self.db.get("requests", "pr", 1)["action"], "rerun")
        self.assertEqual(self.db.find("pr", 1), "reviewed")  # verdict kept


class StartupValidationTests(unittest.TestCase):
    """Broken side configs exit rc=2 at startup, as master's block did."""

    def pr(self, extra: str) -> object:
        import shlex
        return fairy.parse_args(shlex.split("--owner o --repo r " + extra))

    def test_llm_review_cmd_requires_patch_repo(self) -> None:
        ns = self.pr("--llm-review-cmd wrapper")
        with self.assertRaises(SystemExit) as ctx:
            agent.validate_sides(ns, None)
        self.assertEqual(ctx.exception.code, 2)  # rc 2, not 1: cron distinguishes config from crash

    def test_patch_repo_satisfies_the_check(self) -> None:
        agent.validate_sides(self.pr("--llm-review-cmd wrapper --patch-repo p"),
                             None)

    def test_simulate_past_template_must_contain_number(self) -> None:
        base = "--simulate-past 2026-07-01T00:00:00+00:00 "
        with self.assertRaises(SystemExit):
            agent.validate_sides(self.pr(base), None)
        with self.assertRaises(SystemExit):
            agent.validate_sides(
                self.pr(base + "--patch-pr-ref-template fforge/pr/"), None)
        agent.validate_sides(
            self.pr(base + "--patch-pr-ref-template fforge/pr/{number}"), None)

    def test_forced_only_requires_a_force_review(self) -> None:
        with self.assertRaises(SystemExit):
            agent.validate_sides(self.pr("--forced-only"), None)
        agent.validate_sides(self.pr("--forced-only --force-review-pr 5"), None)


class RequestsPassTests(AgentCase):
    """An r/f press must not rescan the whole forge (production: one
    request cost a ~74s full pass); the requests pass fetches only the
    named items."""

    def test_only_the_requested_item_is_fetched(self) -> None:
        self.db.push("requests", "pr", 7, {"action": "rerun"})
        args = agent.parse_args(["--pr-args", "x"])
        with mock.patch.object(fairy, "list_open_prs",
                               side_effect=AssertionError("full listing")), \
                mock.patch.object(fairy, "get_pr",
                                  side_effect=lambda ns, n: make_pr(n)), \
                mock.patch.object(fairy, "safe_prepare_pr", self.prepare), \
                mock.patch.object(forge_gcli, "self_login",
                                  return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.requests_pass(self.db, self.ns, None, args)
        self.assertEqual(self.db.find("pr", 7), "queued")
        self.assertIsNone(self.db.get("requests", "pr", 7))
        self.assertFalse(self.ns.forced_only)  # the clone flips, not ours


class ColorTests(unittest.TestCase):
    def test_side_color_reaches_setup_logging(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        argv = ["agent.py", "--db-root", tmp.name,
                "--pr-args", "--owner o --repo r --color never"]
        with mock.patch.object(agent, "setup_logging") as logging_setup, \
                mock.patch.object(agent, "one_pass"), \
                mock.patch.object(sys, "argv", argv):
            agent.main()
        self.assertEqual(logging_setup.call_args.kwargs["color"], "never")


class _StopLoop(BaseException):
    """Sentinel to end a daemon loop; a BaseException so the loop's own
    ``except Exception`` cannot swallow it."""


class LoopTests(unittest.TestCase):
    """--loop N is the daemon contract: keep passing, and survive a
    failed pass. Without it one pass runs and an error is fatal, so
    cron sees the failure in the exit code."""

    def run_main(self, flags: str, outcomes: list) -> list[int]:
        import shlex
        import time
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        argv = ["agent.py", "--db-root", tmp.name,
                "--pr-args", "--owner o --repo r"] + shlex.split(flags)
        calls = self.calls = []

        def one_pass(*args, **kwargs) -> None:
            calls.append(len(calls))
            outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if outcome is not None:
                raise outcome

        wake = mock.Mock()
        wake.wait.side_effect = lambda timeout=None: time.sleep(timeout or 0)
        with mock.patch.object(agent, "one_pass", side_effect=one_pass), \
                mock.patch.object(agent, "send_pass"), \
                mock.patch.object(agent, "setup_logging"), \
                mock.patch.object(agent, "watch_paths"), \
                mock.patch.object(agent, "Event", return_value=wake), \
                mock.patch.object(sys, "argv", argv):
            self.rc = agent.main()
        return calls

    def test_loop_keeps_scanning(self) -> None:
        with self.assertRaises(_StopLoop):
            self.run_main("--loop 0.01", [None, None, _StopLoop()])
        self.assertEqual(len(self.calls), 3)

    def test_without_loop_a_single_pass_returns(self) -> None:
        self.assertEqual(self.run_main("", [None]), [0])
        self.assertEqual(self.rc, 0)

    def test_a_failed_pass_does_not_kill_the_daemon(self) -> None:
        """A transient forge/gcli error costs one interval, not the
        whole service."""
        with self.assertRaises(_StopLoop):
            self.run_main("--loop 0.01", [RuntimeError("forge 500"),
                                          _StopLoop()])
        self.assertEqual(len(self.calls), 2)

    def test_a_failed_pass_is_fatal_in_one_shot_mode(self) -> None:
        with self.assertRaises(RuntimeError):
            self.run_main("", [RuntimeError("forge 500")])


class OnePassTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> list[str]:
        import worker
        calls: list[str] = []
        args = agent.parse_args(argv)
        with mock.patch.object(agent, "scan_pass",
                               side_effect=lambda *a, **k: calls.append("scan")), \
                mock.patch.object(worker, "drain",
                                  side_effect=lambda *a, **k: calls.append("drain")), \
                mock.patch.object(agent, "send_pass",
                                  side_effect=lambda *a, **k: calls.append("send")), \
                mock.patch.object(agent, "log_summary"):
            agent.one_pass(mock.Mock(), mock.Mock(), None, args)
        return calls

    def test_drain_runs_the_worker_between_scan_and_send(self) -> None:
        self.assertEqual(self._run(["--pr-args", "x", "--drain"]),
                         ["scan", "drain", "send"])

    def test_without_drain_the_agent_never_reviews(self) -> None:
        self.assertEqual(self._run(["--pr-args", "x"]), ["scan", "send"])

    def test_ask_gets_an_inline_retry_only_with_drain(self) -> None:
        retries = []
        with mock.patch.object(agent, "ask_pass",
                               side_effect=lambda db, kinds, retry=None:
                               retries.append(retry)):
            self._run(["--pr-args", "x", "--ask", "--drain"])
            self._run(["--pr-args", "x", "--ask"])
        self.assertIsNotNone(retries[0])
        self.assertIsNone(retries[1])


if __name__ == "__main__":
    unittest.main()


class FinishRequestsUnderClaimTests(AgentCase):
    def test_request_is_consumed_while_the_worker_holds_the_item(self) -> None:
        """Production pr #23914: the lease made the request survive
        every pass, re-forcing the item and destroying each fresh
        verdict."""
        self.db.push("requests", "pr", 5, {"action": "rerun"})
        self.db.push("queued", "pr", 5, {"title": "t"})
        claim = self.db.claim("queued", "llm", "pr", 5)
        try:
            agent.finish_requests(self.db, {"pr": {5}, "issue": set()},
                                  {"pr"})
            self.assertIsNone(self.db.get("requests", "pr", 5))
        finally:
            claim.abort()
