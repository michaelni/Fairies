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

A suspect container halts the run and is never retried.

Regression: the wrapper flagged a review container as suspect (2026-07-28,
PR #23750) and exited 1, which the retry policy could not tell apart from
a transient failure -- so fairy re-ran the same PR against fresh
containers 5s later, while the run's other reviewers kept driving
containers for another 4.5 minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import shell_tool  # noqa: E402
from common import (EXIT_REVIEW_CANCELLED, EXIT_REVIEW_HALTED,  # noqa: E402
                    EXIT_TURN_FAILED, format_turn_failure)


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

    def test_cancelled_review_is_not_retried(self) -> None:
        calls = []

        def invoke(extra_cmd_args):
            calls.append(extra_cmd_args)
            raise fairy.ReviewCancelled("operator cancelled")

        with self.assertRaises(fairy.ReviewCancelled):
            fairy.call_llm_with_retries(_args(3), 23750, invoke)
        self.assertEqual(1, len(calls))

    def test_turn_failed_review_is_not_retried(self) -> None:
        calls = []

        def invoke(extra_cmd_args):
            calls.append(extra_cmd_args)
            raise fairy.ReviewTurnFailed("in-run retry budget spent")

        with self.assertRaises(fairy.ReviewTurnFailed):
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


class TurnFailedErrorPropagationTests(unittest.TestCase):
    """The wrapper's EXIT_TURN_FAILED carries the provider's own error.
    """

    PROVIDER_ERROR = (
        'codex:gpt-5.6-sol: codex exec produced no final message (rc=1); '
        'errors: {"type": "error", "message": "This content was flagged '
        'for possible cybersecurity risk. If this seems wrong, try '
        'rephrasing your request. To get authorized for security work, '
        'join the Trusted Access for Cyber program: '
        'https://chatgpt.com/cyber"}'
    )

    def test_provider_error_reaches_the_exception(self) -> None:
        args = argparse.Namespace(
            llm_review_cmd="./pr_review_wrapper.py", verbose=0,
            llm_timeout=10)
        cp = subprocess.CompletedProcess(
            [], EXIT_TURN_FAILED,
            stdout=format_turn_failure(RuntimeError(self.PROVIDER_ERROR)) + "\n",
            stderr="")
        with mock.patch.object(fairy, "run_cmd", return_value=cp):
            with self.assertRaises(fairy.ReviewTurnFailed) as caught:
                fairy.invoke_llm_wrapper(
                    args, {}, number=42,
                    allowed_classifications=frozenset(),
                    label_allowlist=[])
        self.assertIn(self.PROVIDER_ERROR, str(caught.exception))
        self.assertIn("in-run retry budget", str(caught.exception))


class CancelledExitPropagationTests(unittest.TestCase):
    def test_cancelled_exit_becomes_review_cancelled(self) -> None:
        args = argparse.Namespace(
            llm_review_cmd="./pr_review_wrapper.py", verbose=0,
            llm_timeout=10)
        cp = subprocess.CompletedProcess(
            [], EXIT_REVIEW_CANCELLED, stdout="", stderr="")
        with mock.patch.object(fairy, "run_cmd", return_value=cp):
            with self.assertRaises(fairy.ReviewCancelled):
                fairy.invoke_llm_wrapper(
                    args, {}, number=42,
                    allowed_classifications=frozenset(),
                    label_allowlist=[])


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
