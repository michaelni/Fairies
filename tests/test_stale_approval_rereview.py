"""A stale self-approval must not park the PR on the already-approved skip.

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

import gcli_cache  # noqa: E402
import fairy  # noqa: E402

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
        # The CI-status fetch lies just past the already-approved gate;
        # reaching it proves the stale approval no longer skips the PR.
        with patch.object(
            fairy, "list_commit_statuses", side_effect=_PastGate(),
        ):
            with self.assertRaises(_PastGate):
                _call(REVIEW)

    def test_fresh_approval_still_skips(self) -> None:
        # Same review before the force-pushes (stale not yet set).
        with patch.object(
            fairy, "get_auto_merge_info", return_value="-",
        ):
            decision = _call({**REVIEW, "stale": False})
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, f"already approved by {SELF}")


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
