"""A halted review is never retried.

Regression: the wrapper flagged a review container as suspect (2026-07-28,
PR #23750) and exited 1, which the retry policy could not tell apart from
a transient failure -- so fairy re-ran the same PR against fresh
containers 5s later. A suspect container must stop the PR instead.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


def _args(attempts: int) -> argparse.Namespace:
    return argparse.Namespace(
        llm_max_attempts=attempts, llm_retry_delay=0.0, llm_review_cmd="")


class ReviewHaltedRetryPolicyTests(unittest.TestCase):
    def test_halted_review_is_not_retried(self) -> None:
        calls = []

        def invoke(extra_cmd_args):
            calls.append(extra_cmd_args)
            raise fairy.ReviewHalted("container is suspect")

        with self.assertRaises(fairy.ReviewHalted):
            fairy.call_llm_with_retries(_args(3), 23750, invoke)
        self.assertEqual(1, len(calls))

    def test_ordinary_failure_still_retried(self) -> None:
        calls = []

        def invoke(extra_cmd_args):
            calls.append(extra_cmd_args)
            raise RuntimeError("transient")

        with self.assertRaises(RuntimeError):
            fairy.call_llm_with_retries(_args(3), 23750, invoke)
        self.assertEqual(3, len(calls))


if __name__ == "__main__":
    unittest.main()
