"""start_review_pipeline: --limit caps LLM evaluations.

Mirrors tests/test_issue_fairy.py PipelineTests for the PR pipeline.
"""
import argparse
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import SimpleQueue
from unittest import mock

import fairy
import workset


def make_args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        limit=0,
        llm_parallelism=1,
        cache="/nonexistent",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class CommitStatusCutoffTests(unittest.TestCase):
    def test_simulate_past_filters_post_cutoff_statuses(self) -> None:
        rows = [
            {"context": "old", "created_at": "2026-07-12T00:00:00Z"},
            {"context": "new", "created_at": "2026-07-13T11:22:22Z"},
        ]
        args = argparse.Namespace(owner="o", repo="r",
                                  simulate_past=datetime(2026, 7, 13, tzinfo=timezone.utc))
        with mock.patch.object(fairy, "gcli_api", return_value=rows):
            kept = fairy.list_commit_statuses(args, "sha")
        self.assertEqual(["old"], [s["context"] for s in kept])
        args.simulate_past = None
        with mock.patch.object(fairy, "gcli_api", return_value=rows):
            kept = fairy.list_commit_statuses(args, "sha")
        self.assertEqual(2, len(kept))


def make_pr(n: int) -> dict:
    return {
        "number": n, "title": "t", "user": {"login": "a"},
        "updated_at": "2026-07-19T10:00:00Z", "head": {"sha": "h1"},
        "html_url": f"https://forge/pr/{n}",
    }


class PipelineDriver:
    """start_review_pipeline with prepare and LLM review stubbed out."""

    def _patched(self, fake_review=None):
        def fake_prepare(a, pr, **kw):
            return fairy.PreparedPR(
                pr=pr, number=pr["number"], title="t", author="a",
                auto_merge="-", last_activity=None, base_reason="review",
                discussion=[], reviewer_username="fairy",
            )

        def default_review(a, p):
            return fairy.Decision(p.number, p.title, p.author, "-", "comment",
                                  "llm", None, "reply", "m")

        return (
            mock.patch.object(fairy, "prepare_pr", side_effect=fake_prepare),
            mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                              side_effect=fake_review or default_review),
            mock.patch.object(fairy.gcli_cache, "save_cache"),
        )

    def _start(self, args: argparse.Namespace, numbers: list[int],
               cancelled: set[int] | None = None):
        input_queue: SimpleQueue = SimpleQueue()
        for n in numbers:
            input_queue.put(make_pr(n))
        return input_queue, fairy.start_review_pipeline(
            args, input_queue,
            now=datetime.now(timezone.utc), self_login="fairy",
            wip_re=re.compile("wip"), cache=mock.Mock(),
            discussion_cache_max_age=timedelta(hours=1),
            cancelled=cancelled,
        )

    def _run(self, args: argparse.Namespace, numbers: list[int],
             cancelled: set[int] | None = None, fake_review=None) -> list:
        p1, p2, p3 = self._patched(fake_review)
        with p1, p2, p3:
            input_queue, (reviewed_queue, llm_queue) = self._start(
                args, numbers, cancelled)
            input_queue.put(fairy._PREPARE_DONE)
            results = [reviewed_queue.get(timeout=10) for _ in numbers]
            for _ in range(max(1, args.llm_parallelism)):
                llm_queue.put(fairy._LLM_REVIEW_DONE)
        return results


class PipelineLimitTests(PipelineDriver, unittest.TestCase):
    def test_limit_zero_evaluates_every_pr(self) -> None:
        results = self._run(make_args(llm_parallelism=2, limit=0), [1, 2, 3, 4])
        self.assertEqual(sorted(d.pr_number for _, d in results), [1, 2, 3, 4])
        self.assertTrue(all(d.llm_classification == "reply" for _, d in results))

    def test_limit_caps_llm_evaluations(self) -> None:
        results = self._run(make_args(llm_parallelism=2, limit=2), [1, 2, 3, 4, 5])
        evaluated = [d for _, d in results if d.llm_classification == "reply"]
        skipped = [d for _, d in results if "--limit" in d.reason]
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(len(skipped), 3)

    def test_candidate_injected_after_start_is_prepared(self) -> None:
        p1, p2, p3 = self._patched()
        with p1, p2, p3:
            input_queue, (reviewed_queue, llm_queue) = self._start(
                make_args(), [1])
            self.assertEqual(reviewed_queue.get(timeout=10)[1].pr_number, 1)
            # Force-add while the pipeline is already running (TUI "f" key).
            input_queue.put({"number": 2, "title": "t", "user": {"login": "a"}})
            input_queue.put(fairy._PREPARE_DONE)
            self.assertEqual(reviewed_queue.get(timeout=10)[1].pr_number, 2)
            llm_queue.put(fairy._LLM_REVIEW_DONE)

    def test_cancelled_number_skips_llm_call(self) -> None:
        p1, p2, p3 = self._patched()
        with p1, p2 as review_mock, p3:
            input_queue, (reviewed_queue, llm_queue) = self._start(
                make_args(), [1, 2], cancelled={2})
            input_queue.put(fairy._PREPARE_DONE)
            results = [reviewed_queue.get(timeout=10) for _ in range(2)]
            llm_queue.put(fairy._LLM_REVIEW_DONE)
        by_number = {d.pr_number: d for _, d in results}
        self.assertEqual(by_number[2].reason, "cancelled by operator")
        self.assertEqual(by_number[2].action, "skip")
        self.assertEqual([c.args[1].number for c in review_mock.call_args_list], [1])


class WorksetWriteTests(PipelineDriver, unittest.TestCase):
    """The pipeline persists one JSON work file per LLM-evaluated PR."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.args = make_args(
            workset_dir=Path(self._tmp.name), owner="o", repo="r",
            forge_type="gitea", gcli_account=None,
        )

    def _item(self, n: int) -> workset.WorkItem | None:
        return workset.load_item(fairy.workset_path(self.args, "pr", n))

    def test_reviewed_file_carries_verdict_and_guard(self) -> None:
        self._run(self.args, [1])
        item = self._item(1)
        assert item is not None
        self.assertEqual(item.state, workset.WorkState.REVIEWED)
        self.assertEqual(item.review.classification, "reply")
        self.assertEqual(item.review.message, "m")
        self.assertEqual(item.expected_updated_at, "2026-07-19T10:00:00Z")
        self.assertEqual(item.expected_head_ref, "h1")
        self.assertEqual(item.html_url, "https://forge/pr/1")

    def test_llm_error_lands_in_error_state(self) -> None:
        def failing_review(a, p):
            return fairy.Decision(p.number, p.title, p.author, "-", "error",
                                  "LLM exploded", None, "error", "")

        self._run(self.args, [1], fake_review=failing_review)
        item = self._item(1)
        assert item is not None
        self.assertEqual(item.state, workset.WorkState.ERROR)
        self.assertEqual(item.error, "LLM exploded")
        self.assertIsNone(item.review)

    def test_cancelled_number_lands_in_cancelled_state(self) -> None:
        self._run(self.args, [1], cancelled={1})
        item = self._item(1)
        assert item is not None
        self.assertEqual(item.state, workset.WorkState.CANCELLED)

    def test_operator_deleted_file_stays_deleted(self) -> None:
        def deleting_review(a, p):
            fairy.workset_path(self.args, "pr", p.number).unlink()
            return fairy.Decision(p.number, p.title, p.author, "-", "comment",
                                  "llm", None, "reply", "m")

        self._run(self.args, [1], fake_review=deleting_review)
        self.assertIsNone(self._item(1))


class WorksetReuseTests(unittest.TestCase):
    """A persisted REVIEWED file with a matching guard replaces the LLM call."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.args = make_args(
            workset_dir=Path(self._tmp.name), owner="o", repo="r",
            forge_type="gitea", gcli_account=None, force_review_prs=set(),
        )
        self.prepared = fairy.PreparedPR(
            pr=make_pr(1), number=1, title="t", author="a", auto_merge="-",
            last_activity=None, base_reason="review", discussion=[],
            reviewer_username="fairy",
        )

    def _seed(self, classification: str = "moderate_issues",
              updated_at: str = "2026-07-19T10:00:00Z") -> None:
        now = datetime.now(timezone.utc).isoformat()
        workset.save_item(fairy.workset_path(self.args, "pr", 1), workset.WorkItem(
            kind="pr", forge_type="gitea", account="", owner="o", repo="r",
            number=1, state=workset.WorkState.REVIEWED,
            created_at=now, state_changed_at=now,
            expected_updated_at=updated_at, expected_head_ref="h1",
            review=workset.ReviewResult(
                classification=classification, message="persisted body"),
        ))

    def _evaluate(self):
        with mock.patch.object(fairy, "apply_llm_review_to_prepared") as llm:
            llm.return_value = fairy.Decision(
                1, "t", "a", "-", "comment", "llm", None, "reply", "fresh")
            decision = fairy.safe_apply_llm_review_to_prepared(
                self.args, self.prepared)
        return decision, llm

    def test_guard_match_reuses_without_llm_call(self) -> None:
        self._seed()
        decision, llm = self._evaluate()
        llm.assert_not_called()
        self.assertEqual(decision.llm_message, "persisted body")
        self.assertEqual(decision.action, "comment")

    def test_guard_mismatch_reruns_llm(self) -> None:
        self._seed(updated_at="2026-07-01T00:00:00Z")
        decision, llm = self._evaluate()
        llm.assert_called_once()
        self.assertEqual(decision.llm_message, "fresh")

    def test_persisted_skip_is_not_reused(self) -> None:
        self._seed(classification="skip")
        _, llm = self._evaluate()
        llm.assert_called_once()

    def test_forced_number_reruns_llm(self) -> None:
        self._seed()
        self.args.force_review_prs = {1}
        _, llm = self._evaluate()
        llm.assert_called_once()


class WorksetOperatorEditTests(unittest.TestCase):
    """apply_decision re-reads the item file: edits win, deletion vetoes."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.args = make_args(
            workset_dir=Path(self._tmp.name), owner="o", repo="r",
            forge_type="gitea", gcli_account=None,
        )
        self.prepared = fairy.PreparedPR(
            pr=make_pr(1), number=1, title="t", author="a", auto_merge="-",
            last_activity=None, base_reason="review", discussion=[],
            reviewer_username="fairy",
        )
        self.decision = fairy.Decision(
            1, "t", "a", "-", "comment", "llm", None, "reply", "original")

    def _seed(self, message: str = "original",
              state: workset.WorkState = workset.WorkState.REVIEWED) -> None:
        now = datetime.now(timezone.utc).isoformat()
        workset.save_item(fairy.workset_path(self.args, "pr", 1), workset.WorkItem(
            kind="pr", forge_type="gitea", account="", owner="o", repo="r",
            number=1, state=state, created_at=now, state_changed_at=now,
            review=workset.ReviewResult(classification="reply", message=message),
        ))

    def _apply(self):
        counts = {action: 0 for action in fairy.ACTIONABLE_DECISIONS}
        with mock.patch.object(fairy, "check_pr_still_unchanged", return_value=None), \
                mock.patch.object(fairy, "post_issue_comment") as post:
            fairy.apply_decision(
                self.args, self.prepared, self.decision,
                cache=None, submitted_counts=counts,
            )
        return post

    def test_edited_message_is_what_gets_posted(self) -> None:
        self._seed(message="operator-corrected body")
        post = self._apply()
        self.assertEqual(post.call_args.args[4], "operator-corrected body")
        item = workset.load_item(fairy.workset_path(self.args, "pr", 1))
        self.assertEqual(item.state, workset.WorkState.POSTED)

    def test_deleted_file_vetoes_the_post(self) -> None:
        post = self._apply()
        post.assert_not_called()

    def test_cancelled_file_vetoes_the_post(self) -> None:
        self._seed(state=workset.WorkState.CANCELLED)
        post = self._apply()
        post.assert_not_called()

    def test_guard_suppressed_labels_do_not_mark_posted(self) -> None:
        # regression: the label-only path used to flip the file to POSTED
        label = fairy.LabelChange("needs docs", "add")
        self.decision = fairy.Decision(
            1, "t", "a", "-", "skip", "llm", None, "reply", "original",
            label_changes=(label,))
        self._seed_with_label()
        counts = {action: 0 for action in fairy.ACTIONABLE_DECISIONS}
        with mock.patch.object(fairy, "check_pr_still_unchanged",
                               return_value="PR updated_at changed"):
            fairy.apply_decision(
                self.args, self.prepared, self.decision,
                cache=None, submitted_counts=counts,
            )
        item = workset.load_item(fairy.workset_path(self.args, "pr", 1))
        self.assertEqual(item.state, workset.WorkState.REVIEWED)

    def test_applied_labels_mark_posted(self) -> None:
        label = fairy.LabelChange("needs docs", "add")
        self.decision = fairy.Decision(
            1, "t", "a", "-", "skip", "llm", None, "reply", "original",
            label_changes=(label,))
        self._seed_with_label()
        counts = {action: 0 for action in fairy.ACTIONABLE_DECISIONS}
        with mock.patch.object(fairy, "check_pr_still_unchanged", return_value=None), \
                mock.patch.object(fairy, "get_pr", return_value={"labels": []}), \
                mock.patch.object(fairy, "apply_issue_label_changes"), \
                mock.patch.object(fairy, "post_label_explanations"):
            fairy.apply_decision(
                self.args, self.prepared, self.decision,
                cache=None, submitted_counts=counts,
            )
        item = workset.load_item(fairy.workset_path(self.args, "pr", 1))
        self.assertEqual(item.state, workset.WorkState.POSTED)

    def _seed_with_label(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        workset.save_item(fairy.workset_path(self.args, "pr", 1), workset.WorkItem(
            kind="pr", forge_type="gitea", account="", owner="o", repo="r",
            number=1, state=workset.WorkState.REVIEWED,
            created_at=now, state_changed_at=now,
            review=workset.ReviewResult(
                classification="reply", message="original",
                label_changes=[workset.LabelChange(label="needs docs", op="add")]),
        ))

    def test_non_llm_decision_passes_through(self) -> None:
        # gate decisions (llm "-") never had a file
        self.decision = fairy.Decision(1, "t", "a", "-", "approve", "rules", None)
        counts = {action: 0 for action in fairy.ACTIONABLE_DECISIONS}
        with mock.patch.object(fairy, "check_pr_still_unchanged", return_value=None), \
                mock.patch.object(fairy, "gcli_approve") as approve:
            fairy.apply_decision(
                self.args, self.prepared, self.decision,
                cache=None, submitted_counts=counts,
            )
        approve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
