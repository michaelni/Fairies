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

Automatic reviews of fairy's own pull request stop once fairy approved
it or posted on it twice; a forced review still goes through. Fairy's
posts on its own PR are issue comments, as the forge refuses approving
reviews from the author (FFmpeg #24798: fairy's PR, reviewed by
kasper93).
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402

SELF = "Forgejo_Fairy"
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)

PR = {
    "number": 24798,
    "state": "open",
    "mergeable": True,
    "title": "avfilter/buffersrc: reject invalid video pixel formats",
    "user": {"login": SELF},
    "head": {"sha": "0b4c1f2a6d8e9c3b5a7f1e2d4c6b8a0f9e7d5c3b",
             "ref": "fairy/issue23991-bugfix"},
    "updated_at": "2026-09-29T21:14:46Z",
}


def _comment(login: str, at: str, body: str) -> dict:
    return {"user": {"login": login}, "created_at": at, "updated_at": at,
            "body": body}


HUMAN_REVIEW = _comment(
    "kasper93", "2026-09-29T21:14:46Z",
    "I would prefer if you tell Fairy to split changes in commits.")
FAIRY_POSTS = [
    _comment(SELF, "2026-09-28T10:00:00Z", "Reviewed my own pull request."),
    _comment(SELF, "2026-09-29T12:00:00Z", "Split the test from the fix."),
]


def _call(comments: list[dict], *, posted_classification: str | None = None,
          force_review: frozenset[int] = frozenset()) -> fairy.Decision | fairy.PreparedPR:
    args = SimpleNamespace(
        force_skip_prs=frozenset(), force_review_prs=force_review,
        force_review_non_open=False, force_engage=False,
        owner="FFmpeg", repo="FFmpeg", simulate_past=None, verbose=False,
        llm_review_cmd="./wrapper", include_self_approved=False,
        min_age_days=0.25,
    )
    with (
        patch.object(fairy, "get_auto_merge_info", return_value="no"),
        patch.object(fairy, "list_commit_statuses", return_value=[]),
    ):
        return fairy.prepare_pr(
            args, PR, now=NOW, self_login=SELF,
            wip_re=fairy.compile_wip_regex([]),
            thread=([], comments, [], []),
            posted_classification=posted_classification)


class OwnPullRequestGuardTests(unittest.TestCase):
    def test_first_review_after_human_feedback_goes_ahead(self) -> None:
        self.assertIsInstance(_call([FAIRY_POSTS[0], HUMAN_REVIEW]), fairy.PreparedPR)

    def test_two_own_posts_stop_automatic_reviews(self) -> None:
        decision = _call([*FAIRY_POSTS, HUMAN_REVIEW])
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "own pull request, already reviewed 2 times")

    def test_posted_approval_stops_automatic_reviews(self) -> None:
        decision = _call([FAIRY_POSTS[0], HUMAN_REVIEW],
                         posted_classification="minor_issues_approve")
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "own pull request, already approved")

    def test_forced_review_bypasses_the_guard(self) -> None:
        prepared = _call([*FAIRY_POSTS, HUMAN_REVIEW], force_review=frozenset({24798}),
                         posted_classification="approve")
        self.assertIsInstance(prepared, fairy.PreparedPR)

    def test_foreign_pull_request_is_not_guarded(self) -> None:
        pr_of_alice = {**PR, "user": {"login": "alice"}}
        with patch.dict(PR, pr_of_alice):
            prepared = _call([*FAIRY_POSTS, HUMAN_REVIEW],
                             posted_classification="approve")
        self.assertIsInstance(prepared, fairy.PreparedPR)


if __name__ == "__main__":
    unittest.main()
