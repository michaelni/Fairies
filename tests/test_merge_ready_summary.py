"""Tests for the "merge-ready" reminder lists in the end-of-run summary.

The bot has no API to merge PRs autonomously, so when a PR has been
approved (by the bot or by anyone else), is mergeable, and is not
already queued for auto-merge, the only remaining step is a human
clicking "Merge" in the Forgejo UI. Two disjoint summary sections
surface this:

1. ``merge_ready`` -- bot has approved the PR. Set at the existing
   "already approved by self_login" skip gate.
2. ``external_approvers`` -- a non-bot reviewer has approved and the
   bot is not acting this run. Computed once near the top of
   ``prepare_pr`` (right after ``last_activity``) so it is available
   even on PRs that early-skip via the LLM skip-backoff or the
   forced-review paths.

These tests pin the dataclass shape and the summary filter logic;
the gates themselves are short if-statements that are easier to
read in code than to exercise via mocked ``prepare_pr`` calls.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


def _decision(
    pr_number: int = 42,
    *,
    merge_ready: bool = False,
    external_approvers: tuple[str, ...] = (),
    action: str = "skip",
    title: str = "title",
) -> fairy.Decision:
    return fairy.Decision(
        pr_number=pr_number,
        title=title,
        author="alice",
        auto_merge="-",
        action=action,
        reason="already approved by bot",
        last_activity=datetime(2026, 5, 1, tzinfo=timezone.utc),
        merge_ready=merge_ready,
        external_approvers=external_approvers,
    )


class DecisionFieldTests(unittest.TestCase):
    def test_merge_ready_defaults_to_false(self) -> None:
        d = _decision()
        self.assertFalse(d.merge_ready)

    def test_merge_ready_propagates(self) -> None:
        d = _decision(merge_ready=True)
        self.assertTrue(d.merge_ready)

    def test_external_approvers_defaults_to_empty(self) -> None:
        d = _decision()
        self.assertEqual(d.external_approvers, ())

    def test_external_approvers_propagates(self) -> None:
        d = _decision(external_approvers=("james", "alice"))
        self.assertEqual(d.external_approvers, ("james", "alice"))


def _filter_merge_ready(
    decisions: list[fairy.Decision],
) -> list[fairy.Decision]:
    return sorted(
        (d for d in decisions if d.merge_ready and d.pr_number >= 0),
        key=lambda x: x.pr_number,
    )


def _filter_externally_approved(
    decisions: list[fairy.Decision],
) -> list[fairy.Decision]:
    return sorted(
        (
            d for d in decisions
            if d.external_approvers
            and not d.merge_ready
            and d.action == "skip"
            and d.pr_number >= 0
        ),
        key=lambda x: x.pr_number,
    )


class MergeReadyFilterTests(unittest.TestCase):
    """Pin the filter logic for the "approved by the bot" list."""

    def test_only_merge_ready_decisions_are_listed(self) -> None:
        d_ready = _decision(pr_number=10, merge_ready=True)
        d_not_ready = _decision(pr_number=11, merge_ready=False)
        result = _filter_merge_ready([d_ready, d_not_ready])
        self.assertEqual([d.pr_number for d in result], [10])

    def test_synthetic_negative_pr_numbers_are_excluded(self) -> None:
        # The codebase uses negative PR numbers for synthetic / error
        # placeholder Decisions that must never reach the user-facing
        # summary.
        d_ready = _decision(pr_number=10, merge_ready=True)
        d_synth = _decision(pr_number=-1, merge_ready=True)
        result = _filter_merge_ready([d_ready, d_synth])
        self.assertEqual([d.pr_number for d in result], [10])

    def test_empty_input_yields_empty_list(self) -> None:
        self.assertEqual(_filter_merge_ready([]), [])

    def test_results_sorted_by_pr_number(self) -> None:
        decisions = [
            _decision(pr_number=42, merge_ready=True),
            _decision(pr_number=7, merge_ready=True),
            _decision(pr_number=99, merge_ready=True),
        ]
        result = _filter_merge_ready(decisions)
        self.assertEqual([d.pr_number for d in result], [7, 42, 99])


class ExternallyApprovedFilterTests(unittest.TestCase):
    """Pin the filter logic for the "approved by an external
    reviewer" list (the second reminder section).
    """

    def test_external_approvers_listed_when_bot_skipped(self) -> None:
        d = _decision(
            pr_number=22965,
            external_approvers=("james",),
            action="skip",
            merge_ready=False,
        )
        result = _filter_externally_approved([d])
        self.assertEqual([x.pr_number for x in result], [22965])

    def test_merge_ready_takes_precedence_disjoint_lists(self) -> None:
        # A PR approved by both bot and external reviewer must appear
        # in the bot list only, never duplicated to the external list.
        d = _decision(
            pr_number=10,
            external_approvers=("james",),
            merge_ready=True,
        )
        self.assertEqual(_filter_merge_ready([d]), [d])
        self.assertEqual(_filter_externally_approved([d]), [])

    def test_active_decisions_are_excluded(self) -> None:
        # If the bot is approving the PR this run (action="approve"),
        # surfacing it as "needs human merge" would be premature: it
        # will reappear next run via the merge_ready path once the
        # bot's approval is recorded.
        d = _decision(
            pr_number=10,
            external_approvers=("james",),
            action="approve",
        )
        self.assertEqual(_filter_externally_approved([d]), [])

    def test_empty_external_approvers_excluded(self) -> None:
        d = _decision(pr_number=10, external_approvers=())
        self.assertEqual(_filter_externally_approved([d]), [])

    def test_synthetic_negative_pr_numbers_are_excluded(self) -> None:
        d = _decision(
            pr_number=-1, external_approvers=("james",),
        )
        self.assertEqual(_filter_externally_approved([d]), [])

    def test_results_sorted_by_pr_number(self) -> None:
        decisions = [
            _decision(pr_number=42, external_approvers=("james",)),
            _decision(pr_number=7, external_approvers=("alice",)),
            _decision(pr_number=99, external_approvers=("bob",)),
        ]
        result = _filter_externally_approved(decisions)
        self.assertEqual([d.pr_number for d in result], [7, 42, 99])


if __name__ == "__main__":
    unittest.main()
