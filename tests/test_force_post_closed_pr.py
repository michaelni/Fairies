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

``--force-review`` + ``--force-review-non-open`` posts to a closed PR.

Regression: a forced review reached the send step but the open-state
guard in ``check_pr_still_unchanged`` blocked the post ("not posted
(PR is no longer open)"), so the expensive LLM pass was wasted
(observed live on #23251). Reviewing a non-open PR is opt-in via
``--force-review-non-open``; with it, the post must go through even once
the PR is closed -- the forge still accepts comments there. Without it,
a bare forced review leaves a now-closed PR alone, matching the
prepare-time gate.

The ``updated_at``/head staleness checks must stay active for forced
posts so a review is still pinned to the state it was written against.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402

UPDATED_AT = "2026-06-02T03:00:00Z"
HEAD = "deadbeef"
NUMBER = 23251


def _closed_pr(**overrides: object) -> dict:
    return {
        "number": NUMBER,
        "state": "closed",
        "updated_at": UPDATED_AT,
        "head": {"sha": HEAD},
        **overrides,
    }


def _decision() -> fairy.Decision:
    return fairy.Decision(
        NUMBER, "t", "a", "-", "request_changes", "llm", None,
        expected_pr_updated_at=UPDATED_AT, expected_head_ref=HEAD,
    )


def _check(force: bool, current: dict, *, non_open: bool = False) -> str | None:
    args = SimpleNamespace(
        force_review_prs={NUMBER} if force else set(),
        force_review_non_open=non_open,
    )
    return fairy.check_pr_still_unchanged(args, current, _decision())


class SubmitSeamTests(unittest.TestCase):
    def test_the_seam_posts_without_consulting_the_forge(self) -> None:
        """The staleness guard is the agent's, run on the item it
        fetched; the seam posts what it is given."""
        args = SimpleNamespace(owner="o", repo="r")
        with patch.object(fairy, "get_pr") as get_pr, \
                patch.object(fairy, "post_issue_comment") as post:
            self.assertIsNone(fairy.submit_decision_action(args, _decision()))
        post.assert_called_once()
        get_pr.assert_not_called()


class ForcePostClosedPrTests(unittest.TestCase):
    def test_unforced_closed_pr_is_blocked(self) -> None:
        self.assertEqual(_check(False, _closed_pr()), "PR is no longer open")

    def test_forced_closed_pr_blocked_without_non_open_opt_in(self) -> None:
        self.assertEqual(_check(True, _closed_pr()), "PR is no longer open")

    def test_forced_closed_pr_posts_with_non_open_opt_in(self) -> None:
        self.assertIsNone(_check(True, _closed_pr(), non_open=True))

    def test_forced_post_still_pinned_to_reviewed_head(self) -> None:
        drifted = _closed_pr(head={"sha": "feedface"})
        self.assertEqual(_check(True, drifted, non_open=True), "PR head changed")


if __name__ == "__main__":
    unittest.main()
