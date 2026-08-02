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

Regression test for ``--simulate-past`` filtering of the auto-merge timeline.

``--simulate-past`` (formerly ``--ignore-after``) was first wired only
through ``filter_activity_after`` for reviews / issue comments / inline
review comments. Auto-merge schedule state was still derived from the
*current* timeline, which meant a sim run for a PR whose author cancelled
auto-merge after the cutoff would observe the cancellation as if it
already existed -- contradicting the very purpose of replaying with a
past view.

This test pins the filter point now applied inside ``get_auto_merge_info``:
when a schedule event is created before the cutoff and a cancel event
after, the filtered derivation must stay at ``"merge"`` (the state the
operator would have observed at the cutoff), and the unfiltered
derivation must collapse to ``"no"`` (today's state).

The auto-merge timeline events used here mirror Forgejo's typed comment
shape (``type``, ``created_at``); see ``forge_gcli.AUTO_MERGE_SCHEDULE_EVENT`` /
``AUTO_MERGE_CANCEL_EVENT`` and the linked source-of-truth comment in
``forge_gcli.py``.
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
import forge_gcli  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _forgejo_auto_merge(timeline):
    """The Forgejo path: state derived from the typed timeline entries."""
    return forge_gcli.auto_merge_state(
        SimpleNamespace(forge_type="gitea"), {}, timeline)


IGNORE_AFTER = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
SCHEDULE_EVENT = {
    "type": "pull_scheduled_merge",
    "created_at": "2026-04-30T10:00:00Z",
}
CANCEL_EVENT = {
    "type": "pull_cancel_scheduled_merge",
    "created_at": "2026-05-02T08:00:00Z",
}


class SimulatePastTimelineTests(unittest.TestCase):
    def test_unfiltered_timeline_reflects_current_state(self) -> None:
        # Sanity: without simulate_past the bot sees both events and the
        # latest (cancel) wins. This isn't the new behavior; pinning it
        # protects against a regression that flips the helper's polarity.
        state = _forgejo_auto_merge(
            [SCHEDULE_EVENT, CANCEL_EVENT]
        )
        self.assertEqual(state, "no")

    def test_filtered_timeline_keeps_pre_cutoff_schedule(self) -> None:
        filtered = fairy.filter_activity_after(
            [SCHEDULE_EVENT, CANCEL_EVENT], IGNORE_AFTER, "created_at"
        )
        self.assertEqual(
            _forgejo_auto_merge(filtered),
            "merge",
        )

    def test_filtered_timeline_drops_only_the_cancel_event(self) -> None:
        # Asserting on the filter itself, not just the derived state, so
        # a future change to ``forge_gcli.auto_merge_state`` cannot
        # mask a regression where the cutoff stops filtering.
        filtered = fairy.filter_activity_after(
            [SCHEDULE_EVENT, CANCEL_EVENT], IGNORE_AFTER, "created_at"
        )
        self.assertEqual(filtered, [SCHEDULE_EVENT])

    def test_no_ignore_after_is_passthrough(self) -> None:
        items = [SCHEDULE_EVENT, CANCEL_EVENT]
        self.assertIs(
            fairy.filter_activity_after(items, None, "created_at"),
            items,
        )


class CommitStatusCutoffTests(unittest.TestCase):
    """Ported verbatim from the deleted test_fairy_pipeline.py (cutover
    audit G1): the CI-status list must not leak post-cutoff results
    into a --simulate-past run."""

    def test_simulate_past_filters_post_cutoff_statuses(self) -> None:
        import argparse
        from unittest import mock
        rows = [
            {"context": "old", "created_at": "2026-07-12T00:00:00Z"},
            {"context": "new", "created_at": "2026-07-13T11:22:22Z"},
        ]
        args = argparse.Namespace(owner="o", repo="r",
                                  simulate_past=datetime(2026, 7, 13, tzinfo=timezone.utc))
        with mock.patch.object(fairy.forge_gcli, "gcli_api", return_value=rows):
            kept = fairy.list_commit_statuses(args, "sha")
        self.assertEqual(["old"], [s["context"] for s in kept])
        args.simulate_past = None
        with mock.patch.object(fairy.forge_gcli, "gcli_api", return_value=rows):
            kept = fairy.list_commit_statuses(args, "sha")
        self.assertEqual(2, len(kept))


if __name__ == "__main__":
    unittest.main()
