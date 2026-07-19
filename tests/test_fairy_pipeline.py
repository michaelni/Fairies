"""start_review_pipeline: --limit caps LLM evaluations.

Mirrors tests/test_issue_fairy.py PipelineTests for the PR pipeline.
"""
import argparse
import re
import unittest
from datetime import datetime, timedelta, timezone
from queue import SimpleQueue
from unittest import mock

import bot_state
import fairy


def make_args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        limit=0,
        llm_parallelism=1,
        cache="/nonexistent",
        fairy_state_cache="/nonexistent",
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


class PipelineLimitTests(unittest.TestCase):
    def _patched(self):
        def fake_prepare(a, pr, **kw):
            return fairy.PreparedPR(
                pr=pr, number=pr["number"], title="t", author="a",
                auto_merge="-", last_activity=None, base_reason="review",
                discussion=[], reviewer_username="fairy",
            )

        def fake_review(a, p, *, state):
            return fairy.Decision(p.number, p.title, p.author, "-", "comment",
                                  "llm", None, "reply", "m")

        return (
            mock.patch.object(fairy, "prepare_pr", side_effect=fake_prepare),
            mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                              side_effect=fake_review),
            mock.patch.object(fairy.gcli_cache, "save_cache"),
            mock.patch.object(fairy.bot_state, "save"),
        )

    def _start(self, args: argparse.Namespace, numbers: list[int],
               cancelled: set[int] | None = None):
        input_queue: SimpleQueue = SimpleQueue()
        for n in numbers:
            input_queue.put({"number": n, "title": "t", "user": {"login": "a"}})
        return input_queue, fairy.start_review_pipeline(
            args, input_queue,
            now=datetime.now(timezone.utc), self_login="fairy",
            wip_re=re.compile("wip"), cache=mock.Mock(),
            state=bot_state.State(),
            discussion_cache_max_age=timedelta(hours=1),
            cancelled=cancelled,
        )

    def _run(self, args: argparse.Namespace, numbers: list[int]) -> list:
        p1, p2, p3, p4 = self._patched()
        with p1, p2, p3, p4:
            input_queue, (reviewed_queue, llm_queue) = self._start(args, numbers)
            input_queue.put(fairy._PREPARE_DONE)
            results = [reviewed_queue.get(timeout=10) for _ in numbers]
            for _ in range(max(1, args.llm_parallelism)):
                llm_queue.put(fairy._LLM_REVIEW_DONE)
        return results

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
        p1, p2, p3, p4 = self._patched()
        with p1, p2, p3, p4:
            input_queue, (reviewed_queue, llm_queue) = self._start(
                make_args(), [1])
            self.assertEqual(reviewed_queue.get(timeout=10)[1].pr_number, 1)
            # Force-add while the pipeline is already running (TUI "f" key).
            input_queue.put({"number": 2, "title": "t", "user": {"login": "a"}})
            input_queue.put(fairy._PREPARE_DONE)
            self.assertEqual(reviewed_queue.get(timeout=10)[1].pr_number, 2)
            llm_queue.put(fairy._LLM_REVIEW_DONE)

    def test_cancelled_number_skips_llm_call(self) -> None:
        p1, p2, p3, p4 = self._patched()
        with p1, p2 as review_mock, p3, p4:
            input_queue, (reviewed_queue, llm_queue) = self._start(
                make_args(), [1, 2], cancelled={2})
            input_queue.put(fairy._PREPARE_DONE)
            results = [reviewed_queue.get(timeout=10) for _ in range(2)]
            llm_queue.put(fairy._LLM_REVIEW_DONE)
        by_number = {d.pr_number: d for _, d in results}
        self.assertEqual(by_number[2].reason, "cancelled by operator")
        self.assertEqual(by_number[2].action, "skip")
        self.assertEqual([c.args[1].number for c in review_mock.call_args_list], [1])


if __name__ == "__main__":
    unittest.main()
