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

Regression tests for API-based auto-merge schedule detection.

The bot used to detect auto-merge state by HTTP-fetching the rendered
PR HTML page and grep-matching localized prose like "scheduled this
pull request to auto merge when all checks succeed". That was both
fragile (one i18n / template tweak in Forgejo silently turned the
result into "no") and incorrect for "scheduled, then canceled, then
re-scheduled" histories (both substrings appeared in the rendered
page, so the scrape returned ``"?"`` instead of the latest state).

The replacement uses the typed ``/issues/{n}/timeline`` events
documented in the Forgejo source:

* ``CommentTypePRScheduledToAutoMerge`` (string ``"pull_scheduled_merge"``)
* ``CommentTypePRUnScheduledToAutoMerge`` (string ``"pull_cancel_scheduled_merge"``)

See ``models/issues/comment.go`` upstream
(https://codeberg.org/forgejo/forgejo/src/branch/forgejo/models/issues/comment.go)
for the constants and ``CreateAutoMergeComment``.

Fixtures under ``tests/fixtures/forgejo_pr_timeline/`` are real,
unmodified responses captured from a live Forgejo instance
(code.ffmpeg.org) so the tests pin against the actual API shape and
not a hand-written approximation:

* ``ffmpeg_pr_22729_timeline.json`` -- PR with auto-merge SCHEDULED
  (one ``pull_scheduled_merge`` event in its timeline).
* ``ffmpeg_pr_21226_timeline.json`` -- PR with NO auto-merge events
  (regular discussion only).
"""

from __future__ import annotations

import json
import sys
import unittest
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

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "forgejo_pr_timeline"


def _load(name: str) -> list[dict]:
    with (FIXTURES / name).open() as f:
        return json.load(f)


class FixtureSanityTests(unittest.TestCase):
    """Pin the captured fixture contents so a future re-capture cannot
    silently turn a "scheduled" PR into a "not-scheduled" one without
    a test failure flagging the change."""

    def test_22729_contains_one_schedule_event(self) -> None:
        timeline = _load("ffmpeg_pr_22729_timeline.json")
        sched = [e for e in timeline if e.get("type") == "pull_scheduled_merge"]
        self.assertEqual(
            len(sched), 1,
            f"expected exactly one pull_scheduled_merge in #22729; "
            f"got {len(sched)}",
        )

    def test_21226_contains_no_auto_merge_events(self) -> None:
        timeline = _load("ffmpeg_pr_21226_timeline.json")
        relevant = [
            e for e in timeline
            if e.get("type") in (
                "pull_scheduled_merge", "pull_cancel_scheduled_merge",
            )
        ]
        self.assertEqual(
            relevant, [],
            f"expected no auto-merge events in #21226; got {relevant}",
        )


class AutoMergeStateFromTimelineTests(unittest.TestCase):
    """Pure-function tests for the timeline-walking classifier."""

    def test_scheduled_pr_resolves_to_merge(self) -> None:
        timeline = _load("ffmpeg_pr_22729_timeline.json")
        self.assertEqual(
            _forgejo_auto_merge(timeline), "merge",
        )

    def test_pr_without_auto_merge_events_resolves_to_no(self) -> None:
        timeline = _load("ffmpeg_pr_21226_timeline.json")
        self.assertEqual(
            _forgejo_auto_merge(timeline), "no",
        )

    def test_empty_timeline_resolves_to_no(self) -> None:
        self.assertEqual(
            _forgejo_auto_merge([]), "no",
        )

    def test_only_unrelated_event_types_resolves_to_no(self) -> None:
        timeline = [
            {"type": "comment", "created_at": "2026-04-01T00:00:00Z"},
            {"type": "label",   "created_at": "2026-04-02T00:00:00Z"},
            {"type": "review",  "created_at": "2026-04-03T00:00:00Z"},
        ]
        self.assertEqual(
            _forgejo_auto_merge(timeline), "no",
        )

    def test_scheduled_then_canceled_resolves_to_no(self) -> None:
        # Latest event is the cancellation -> currently NOT scheduled.
        # The old HTML scrape returned "?" here because both substrings
        # matched; this is the bug-fix the new implementation pins.
        timeline = [
            {"type": "pull_scheduled_merge",
             "created_at": "2026-04-01T10:00:00Z"},
            {"type": "pull_cancel_scheduled_merge",
             "created_at": "2026-04-01T11:00:00Z"},
        ]
        self.assertEqual(
            _forgejo_auto_merge(timeline), "no",
        )

    def test_scheduled_then_canceled_then_rescheduled_resolves_to_merge(
        self,
    ) -> None:
        # Latest of the relevant pair is the most recent schedule, even
        # though a cancellation sits between two schedules.
        timeline = [
            {"type": "pull_scheduled_merge",
             "created_at": "2026-04-01T10:00:00Z"},
            {"type": "pull_cancel_scheduled_merge",
             "created_at": "2026-04-01T11:00:00Z"},
            {"type": "pull_scheduled_merge",
             "created_at": "2026-04-01T12:00:00Z"},
        ]
        self.assertEqual(
            _forgejo_auto_merge(timeline), "merge",
        )

    def test_iso_timestamp_lex_sort_is_correct(self) -> None:
        # ISO-8601 UTC ``Z`` strings sort lexicographically the same as
        # chronologically; this check pins that property so a future
        # refactor that swaps the impl to e.g. integer comparison cannot
        # accidentally drift on day/month boundaries.
        timeline = [
            {"type": "pull_scheduled_merge",
             "created_at": "2026-01-31T23:59:59Z"},
            {"type": "pull_cancel_scheduled_merge",
             "created_at": "2026-02-01T00:00:00Z"},
        ]
        self.assertEqual(
            _forgejo_auto_merge(timeline), "no",
        )

    def test_event_with_missing_created_at_is_skipped(self) -> None:
        # If Forgejo ever emits an event without a created_at string we
        # must not raise; we simply ignore the malformed one and use
        # the rest. Pin that contract so a defensive refactor cannot
        # accidentally promote it to a hard error.
        timeline = [
            {"type": "pull_scheduled_merge"},  # no created_at
            {"type": "pull_cancel_scheduled_merge",
             "created_at": "2026-04-01T11:00:00Z"},
        ]
        self.assertEqual(
            _forgejo_auto_merge(timeline), "no",
        )


if __name__ == "__main__":
    unittest.main()
