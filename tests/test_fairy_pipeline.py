"""start_review_pipeline: --limit caps LLM evaluations.

Mirrors tests/test_issue_fairy.py PipelineTests for the PR pipeline.
"""
import argparse
import re
import unittest
from datetime import datetime, timedelta, timezone
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
    def _run(self, args: argparse.Namespace, numbers: list[int]) -> list:
        prs = [{"number": n, "title": "t", "user": {"login": "a"}} for n in numbers]

        def fake_prepare(a, pr, **kw):
            return fairy.PreparedPR(
                pr=pr, number=pr["number"], title="t", author="a",
                auto_merge="-", last_activity=None, base_reason="review",
                discussion=[], reviewer_username="fairy",
            )

        def fake_review(a, p, *, state):
            return fairy.Decision(p.number, p.title, p.author, "-", "comment",
                                  "llm", None, "reply", "m")

        with mock.patch.object(fairy, "prepare_pr", side_effect=fake_prepare), \
                mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                                  side_effect=fake_review), \
                mock.patch.object(fairy.gcli_cache, "save_cache"), \
                mock.patch.object(fairy.bot_state, "save"):
            reviewed_queue, llm_queue = fairy.start_review_pipeline(
                args, prs,
                now=datetime.now(timezone.utc), self_login="fairy",
                wip_re=re.compile("wip"), cache=mock.Mock(),
                state=bot_state.State(),
                discussion_cache_max_age=timedelta(hours=1),
            )
            results = [reviewed_queue.get(timeout=10) for _ in prs]
            for _ in range(max(1, args.llm_parallelism)):
                llm_queue.put(fairy._LLM_REVIEW_DONE)
        return results

    def test_limit_zero_evaluates_every_pr(self) -> None:
        results = self._run(make_args(llm_parallelism=2, limit=0), [1, 2, 3, 4])
        self.assertEqual(sorted(r.decision.pr_number for r in results), [1, 2, 3, 4])
        self.assertTrue(all(r.decision.llm_classification == "reply" for r in results))

    def test_limit_caps_llm_evaluations(self) -> None:
        results = self._run(make_args(llm_parallelism=2, limit=2), [1, 2, 3, 4, 5])
        evaluated = [r for r in results if r.decision.llm_classification == "reply"]
        skipped = [r for r in results if "--limit" in r.decision.reason]
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(len(skipped), 3)


if __name__ == "__main__":
    unittest.main()
