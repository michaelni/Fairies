"""A suspect container halts the run and is never retried.

Regression: the wrapper flagged a review container as suspect (2026-07-28,
PR #23750) and exited 1, which the retry policy could not tell apart from
a transient failure -- so fairy re-ran the same PR against fresh
containers 5s later, while the run's other reviewers kept driving
containers for another 4.5 minutes.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import shell_tool  # noqa: E402
from common import EXIT_REVIEW_HALTED  # noqa: E402


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


class HaltStopsSiblingReviewersTests(unittest.TestCase):
    def test_halted_shell_call_exits_with_the_halted_status(self) -> None:
        with mock.patch.object(shell_tool, "_halted", False):
            shell_tool.halt()
            with self.assertRaises(SystemExit) as caught:
                shell_tool.exec_shell_call(
                    None, {"command": "true"}, max_timeout_s=1.0)
        # SystemExit so it escapes the codex dispatch loop's except Exception;
        # the status keeps the wrapper's halted exit code intact.
        self.assertEqual(EXIT_REVIEW_HALTED, caught.exception.code)

    def test_not_halted_by_default(self) -> None:
        self.assertFalse(shell_tool.halted())


if __name__ == "__main__":
    unittest.main()
