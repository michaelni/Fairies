"""A silent force-push must count as activity for the re-review gates.

Regression: FFmpeg #22961 (and the wider haasn cohort) sat skipped after
fairy had already left a review *comment* and the author then
force-pushed new commits WITHOUT any accompanying comment. ``last_activity``
was computed from discussion items only (``pr.updated_at`` is too noisy to
trust), so the push was invisible and the "no activity since prior
non-approval message by Forgejo_Fairy" gate froze the PR at fairy's own
last comment forever.

The fix folds the latest ``pull_push`` timeline event into
``last_activity``. These fixtures mirror the live #22961 wire shape:

* fairy's review is an *issue comment* (not an APPROVED review),
* a ``REQUEST_REVIEW`` pseudo-entry (fairy is a requested reviewer),
* a force-push by ``haasn`` dated AFTER fairy's comment.
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bot_state  # noqa: E402
import gcli_cache  # noqa: E402
import fairy  # noqa: E402

SELF = "Forgejo_Fairy"

FAIRY_REVIEW_AT = "2026-06-16T22:21:54Z"
PUSH_AT = "2026-06-17T11:07:39Z"
HEAD_SHA = "eb0ac6b125aead40ac7dd03ee191061a5dd8b687"
PRIOR_SHA = "cbc35cd28b70b0d71d692a1b5cd27a5e26be4b3d"

# 12h after the push: well past a 6h settle window.
NOW = datetime(2026, 6, 17, 23, 8, tzinfo=timezone.utc)

REQUEST_REVIEW = {
    "user": {"id": 852, "login": SELF, "username": SELF},
    "state": "REQUEST_REVIEW",
    "body": "",
    "submitted_at": "2026-06-16T14:48:20Z",
}

FAIRY_COMMENT = {
    "user": {"id": 852, "login": SELF, "username": SELF},
    "created_at": FAIRY_REVIEW_AT,
    "updated_at": FAIRY_REVIEW_AT,
    "body": "LLM review here -- I did an exhaustive pass over all 13 commits ...",
}

PUSH_TIMELINE = [{
    "type": "pull_push",
    "created_at": PUSH_AT,
    "user": {"login": "haasn", "username": "haasn"},
    "body": json.dumps({"is_force_push": True, "commit_ids": [PRIOR_SHA, HEAD_SHA]}),
}]


def _call(*, timeline: list[dict], min_age_days: float = 0.25) -> object:
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
        "number": 22961,
        "state": "open",
        "mergeable": True,
        "title": "Add libavutil/int128.h and libavutil/rational64.h, use in swscale",
        "head": {"sha": HEAD_SHA},
        "updated_at": PUSH_AT,
    }
    with patch.object(
        fairy, "get_pr_discussion",
        return_value=([REQUEST_REVIEW], [FAIRY_COMMENT], []),
    ), patch.object(
        fairy.gcli_cache, "get", return_value={"timeline": timeline},
    ), patch.object(
        fairy, "list_commit_statuses", return_value=[],
    ):
        return fairy.prepare_pr(
            args, pr, now=NOW, self_login=SELF,
            wip_re=fairy.compile_wip_regex([]),
            cache=gcli_cache.Cache(),
            state=bot_state.State(),
            discussion_cache_max_age=timedelta(hours=1),
        )


class PushResetsActivityGateTests(unittest.TestCase):
    def test_force_push_gets_past_non_approval_gate(self) -> None:
        # With the push folded into last_activity, the PR clears the
        # "no activity since prior non-approval" gate and only stops at
        # the empty-CI skip (list_commit_statuses patched to []).
        decision = _call(timeline=PUSH_TIMELINE)
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "no commit statuses / CI results found")
        self.assertEqual(decision.last_activity, fairy.iso_to_dt(PUSH_AT))

    def test_without_push_still_frozen_on_non_approval_gate(self) -> None:
        # No push event -> last_activity stays at fairy's comment ->
        # the non-approval gate correctly skips (unchanged behavior).
        decision = _call(timeline=[])
        self.assertEqual(decision.action, "skip")
        self.assertEqual(
            decision.reason,
            f"no activity since prior non-approval message by {SELF}",
        )


if __name__ == "__main__":
    unittest.main()
