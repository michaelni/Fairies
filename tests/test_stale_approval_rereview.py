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

A stale self-approval must not park the PR on the already-approved skip.

Regression: fairy approved FFmpeg #20148 on 2026-05-03; the author
force-pushed afterwards (05-05, 05-18), so the forge voided the approval
(``stale: true``, web UI shows "0 of 1 approvals granted"). The
already-approved gate only looked at the review *state*, so every run
skipped with "already approved by Forgejo_Fairy" and the PR was never
reconsidered for review.

``REVIEW`` is the real review object served by the forge for #20148
(from the live gcli cache). Forgejo/Gitea set ``stale``/``dismissed``;
GitHub REST reviews have neither field (branch protection rewrites the
state to DISMISSED instead), so absent fields read False there.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent  # noqa: E402
import gcli_cache  # noqa: E402
import fairy  # noqa: E402
import forge_gcli  # noqa: E402

SELF = "Forgejo_Fairy"

# Real review from FFmpeg/FFmpeg #20148 (user profile fields trimmed).
REVIEW = {
    "id": 5262,
    "user": {"id": 852, "login": SELF, "username": SELF},
    "team": None,
    "state": "APPROVED",
    "body": "",
    "commit_id": "14a78b0686b7060d1654db9dd2574c354a21caeb",
    "stale": True,
    "official": True,
    "dismissed": False,
    "comments_count": 0,
    "submitted_at": "2026-05-03T13:27:21Z",
    "updated_at": "2026-05-03T13:27:21Z",
    "html_url": "https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/20148#issuecomment-39182",
    "pull_request_url": "https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/20148",
}

NOW = datetime(2026, 6, 10, tzinfo=timezone.utc)


class _PastGate(Exception):
    """Sentinel: ``prepare_pr`` got past the already-approved gate."""


def _call(review: dict, *, min_age_days: float = 30) -> object:
    args = SimpleNamespace(
        force_skip_prs=frozenset(),
        force_review_prs=frozenset(),
        owner="FFmpeg",
        repo="FFmpeg",
        simulate_past=None,
        verbose=False,
        llm_review_cmd=None,
        include_self_approved=False,
        min_age_days=min_age_days,
    )
    pr = {
        "number": 20148,
        "state": "open",
        "mergeable": True,
        "title": "avdevice/gdigrab: fix -show_region 1 overlay window",
        "head": {"sha": "c9f8d93bf1"},
    }
    with patch.object(
        fairy, "get_pr_discussion", return_value=([review], [], []),
    ):
        return fairy.prepare_pr(
            args, pr, now=NOW, self_login=SELF,
            wip_re=fairy.compile_wip_regex([]),
            cache=gcli_cache.Cache(),
            discussion_cache_max_age=timedelta(hours=1),
        )


class StaleApprovalRereviewTests(unittest.TestCase):
    def test_effective_state_carries_staleness(self) -> None:
        states = fairy.effective_review_states([REVIEW])
        self.assertEqual(states[SELF].state, "APPROVED")
        self.assertTrue(states[SELF].stale)

    def test_dismissed_counts_as_stale(self) -> None:
        review = {**REVIEW, "stale": False, "dismissed": True}
        self.assertTrue(
            fairy.effective_review_states([review])[SELF].stale
        )

    def test_stale_approval_is_reconsidered(self) -> None:
        # The min-age lookup lies just past the already-approved gate;
        # reaching it proves the stale approval no longer skips the PR.
        with patch.object(
            fairy, "effective_min_age_days", side_effect=_PastGate(),
        ), patch.object(fairy, "list_commit_statuses", return_value=[]):
            with self.assertRaises(_PastGate):
                _call(REVIEW)

    def test_fresh_approval_still_skips(self) -> None:
        # Same review before the force-pushes (stale not yet set).
        with patch.object(
            fairy, "get_auto_merge_info", return_value="-",
        ), patch.object(fairy, "list_commit_statuses", return_value=[]):
            decision = _call({**REVIEW, "stale": False})
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, f"already approved by {SELF}")


class ApprovedPrCiTests(unittest.TestCase):
    """An approved PR whose CI waits for a human is not merge-ready.

    The rows are status rows of FFmpeg #24731, which fairy approved on
    request while the forge held its CI for a maintainer's release.
    """

    LINT = "Lint / Pre-Commit (pull_request)"
    FATE = "Test / Fate (linux-amd64, static, 32 bit) (pull_request)"
    PASSED = [
        {"id": 21, "status": "success", "description": "Successful in 10s",
         "target_url": "/FFmpeg/FFmpeg/actions/runs/89916/jobs/0",
         "context": "Autolabel / Labeler (pull_request_target)",
         "created_at": "2026-09-26T22:46:53Z", "updated_at": "2026-09-26T22:46:53Z"},
        {"id": 13, "status": "skipped", "description": "Has been skipped",
         "target_url": "/FFmpeg/FFmpeg/actions/runs/89902/jobs/0",
         "context": "Backport / PR (pull_request_target)",
         "created_at": "2026-09-26T21:38:20Z", "updated_at": "2026-09-26T21:38:20Z"},
    ]
    HELD = [
        {"id": 2, "status": "pending", "description": "Blocked by required conditions",
         "target_url": "/FFmpeg/FFmpeg/actions/runs/89899/jobs/0",
         "context": LINT,
         "created_at": "2026-09-26T21:38:12Z", "updated_at": "2026-09-26T21:38:12Z"},
        {"id": 3, "status": "pending", "description": "Blocked by required conditions",
         "target_url": "/FFmpeg/FFmpeg/actions/runs/89900/jobs/0",
         "context": FATE,
         "created_at": "2026-09-26T21:38:12Z", "updated_at": "2026-09-26T21:38:12Z"},
    ]

    def decide(self, wire: list[dict]) -> fairy.Decision:
        with patch.object(
            fairy, "get_auto_merge_info", return_value="-",
        ), patch.object(
            fairy, "list_commit_statuses",
            return_value=[forge_gcli._project_status_row(r) for r in wire],
        ):
            return _call({**REVIEW, "stale": False})

    def test_ci_held_for_a_human_is_ci_blocked(self) -> None:
        decision = self.decide(self.PASSED + self.HELD)
        self.assertFalse(decision.merge_ready)
        self.assertEqual(decision.blocked_ci_contexts, (self.LINT, self.FATE))
        self.assertEqual(agent.gate_state(decision), "ci-blocked")

    def test_passed_ci_is_merge_ready(self) -> None:
        decision = self.decide(self.PASSED)
        self.assertEqual(agent.gate_state(decision), "merge-ready")


class StaleExternalApprovalTests(unittest.TestCase):
    """Stale external approvals must not show up as "waiting to be merged".

    The forge no longer counts a stale approval towards mergeability, so
    the end-of-run reminder would tell the operator to merge a PR the
    forge refuses to merge. ``min_age_days`` is set high so the run lands
    on the threshold skip, whose Decision carries ``external_approvers``.
    """

    EXT = {"id": 1, "login": "bob", "username": "bob"}

    def test_stale_external_approval_excluded(self) -> None:
        decision = _call({**REVIEW, "user": self.EXT}, min_age_days=365)
        self.assertEqual(decision.reason, "activity is newer than threshold")
        self.assertEqual(decision.external_approvers, ())

    def test_fresh_external_approval_listed(self) -> None:
        with patch.object(
            fairy, "get_auto_merge_info", return_value="-",
        ):
            decision = _call(
                {**REVIEW, "user": self.EXT, "stale": False}, min_age_days=365,
            )
        self.assertEqual(decision.reason, "activity is newer than threshold")
        self.assertEqual(decision.external_approvers, ("bob",))


if __name__ == "__main__":
    unittest.main()
