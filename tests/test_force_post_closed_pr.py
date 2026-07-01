"""``--force-review-pr`` + ``--force-review-non-open`` posts to a closed PR.

Regression: a forced review reached the submit step but the open-state
guard in ``check_pr_still_unchanged`` blocked the post ("submit skipped
because PR is no longer open"), so the expensive LLM pass was wasted
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
    # Used as the ``prepared`` item; ``get_submission_guard`` reads the
    # expected_* fields from it for the staleness checks.
    return fairy.Decision(
        NUMBER, "t", "a", "-", "request_changes", "llm", None,
        expected_pr_updated_at=UPDATED_AT, expected_head_ref=HEAD,
    )


def _check(force: bool, current: dict, *, non_open: bool = False) -> str | None:
    args = SimpleNamespace(
        force_review_prs={NUMBER} if force else set(),
        force_review_non_open=non_open,
    )
    decision = _decision()
    with patch.object(fairy, "get_pr", return_value=current):
        return fairy.check_pr_still_unchanged(args, decision, decision)


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
