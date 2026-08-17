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

Regression tests for ``pull_push`` events surfacing in the LLM
discussion.

Bug history: on 2026-05-26 the triage LLM skipped PR #23197 with the
reason "I already left a blocking review comment on the current head,
and there hasn't been any new author activity or new commit addressing
those points yet." A force-push to the PR head had landed ~3 hours
earlier (``b8cec2c36b87`` -> ``97ca45b95f55``). The triage payload
exposed the new ``head_sha`` and the bot's own prior comment ("I did a
deep pass on b8cec2c36b87 ...") but provided NO direct push-event
signal, so the LLM was forced to speculate by comparing SHAs across
free-form text -- and speculated wrong.

The fix is to pull ``pull_push`` events out of the typed
``/issues/{n}/timeline`` payload and inject them as
``kind="push"`` items into the same discussion list the LLM already
consumes. These tests pin both halves of the fix:

* ``push_events_from_timeline`` extracts the right fields, in the
  right shape, from a real Forgejo timeline (``ffmpeg_pr_23197``).
* ``build_llm_discussion`` merges push items with comments / reviews
  and keeps them in chronological order so the LLM can reason about
  "what happened after my last comment".

Fixture ``ffmpeg_pr_23197_timeline.json`` is an unmodified capture from
code.ffmpeg.org so the test pins the actual Forgejo wire shape, not a
hand-written approximation.
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

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "forgejo_pr_timeline"

# The exact SHAs / timestamps the live PR exhibited; pinned so a future
# re-capture of the fixture cannot silently move the bug under our feet.
PR_23197_INITIAL_SHA = "b8cec2c36b877caf65e8705333993569b97a6a3a"
PR_23197_NEW_HEAD_SHA = "97ca45b95f55c79185e89423d893623ea5a5e52f"
PR_23197_INITIAL_PUSH_AT = "2026-05-21T21:23:41Z"
PR_23197_FORCE_PUSH_AT = "2026-05-25T21:46:48Z"


def _project(events: list[dict]) -> list[dict]:
    """Raw capture -> what ``forge_gcli.list_issue_timeline`` hands out."""
    return [forge_gcli.project_timeline_event(e) for e in events]


def _load_timeline(name: str) -> list[dict]:
    with (FIXTURES / name).open() as f:
        return _project(json.load(f))


class PushEventsFromTimelineTests(unittest.TestCase):

    def test_pr_23197_yields_two_push_events_in_order(self) -> None:
        timeline = _load_timeline("ffmpeg_pr_23197_timeline.json")
        pushes = fairy.push_events_from_timeline(timeline)

        self.assertEqual(
            [p["created_at"] for p in pushes],
            [PR_23197_INITIAL_PUSH_AT, PR_23197_FORCE_PUSH_AT],
        )
        self.assertEqual(
            [p["kind"] for p in pushes], ["push", "push"],
        )
        self.assertEqual(
            [p["author"] for p in pushes], ["michaelni", "michaelni"],
        )
        self.assertEqual(
            [p["is_force_push"] for p in pushes], [False, True],
        )
        # The force-push event carries both the prior head and the new
        # head in ``commit_ids``; we surface the LAST one as ``head_sha``
        # because that is the SHA the LLM will compare against the
        # current ``pull_request.head_sha``.
        self.assertEqual(pushes[0]["head_sha"], PR_23197_INITIAL_SHA)
        self.assertEqual(pushes[1]["head_sha"], PR_23197_NEW_HEAD_SHA)
        self.assertEqual(pushes[0]["commit_count"], 1)
        self.assertEqual(pushes[1]["commit_count"], 2)

    def test_empty_timeline_yields_no_push_events(self) -> None:
        self.assertEqual(fairy.push_events_from_timeline([]), [])

    def test_non_push_events_are_ignored(self) -> None:
        timeline = [
            {"type": "comment", "created_at": "2026-04-01T00:00:00Z"},
            {"type": "label",   "created_at": "2026-04-02T00:00:00Z"},
            {"type": "review",  "created_at": "2026-04-03T00:00:00Z"},
        ]
        self.assertEqual(fairy.push_events_from_timeline(_project(timeline)), [])

    def test_malformed_push_body_is_skipped(self) -> None:
        # If Forgejo ever changes the body encoding we must not crash.
        # Skipping the bad entry and surfacing the rest is the desired
        # degradation (better an incomplete LLM signal than a hard
        # review failure that prevents posting any review at all).
        timeline = [
            {"type": "pull_push", "created_at": "2026-04-01T00:00:00Z",
             "body": "this is not json",
             "user": {"login": "alice"}},
            {"type": "pull_push", "created_at": "2026-04-02T00:00:00Z",
             "body": json.dumps({
                 "is_force_push": False,
                 "commit_ids": ["deadbeef" * 5],
             }),
             "user": {"login": "bob"}},
        ]
        pushes = fairy.push_events_from_timeline(_project(timeline))
        self.assertEqual(len(pushes), 1)
        self.assertEqual(pushes[0]["author"], "bob")
        self.assertEqual(pushes[0]["head_sha"], "deadbeef" * 5)


class BuildLlmDiscussionSortTests(unittest.TestCase):
    def test_edited_comment_sorts_by_creation_not_edit(self) -> None:
        """An edit must not move a comment past a later arrival: the
        list claims chronological order to the LLM prompt, and the
        TUI's sampled separator is placed by counting arrivals in
        list order. Creation/edit stamps from FFmpeg issue #22240."""
        edited = {"user": {"login": "a"}, "body": "edited",
                  "created_at": "2026-02-22T09:29:02Z",
                  "updated_at": "2026-02-22T09:30:32Z"}
        later = {"user": {"login": "b"}, "body": "later",
                 "created_at": "2026-02-22T09:30:00Z",
                 "updated_at": "2026-02-22T09:30:00Z"}
        items = fairy.build_llm_discussion([], [later, edited], [])
        self.assertEqual([i["body"] for i in items], ["edited", "later"])


class BuildLlmDiscussionWithTimelineTests(unittest.TestCase):
    """Pin that the push events show up in the discussion list AND in
    chronological order relative to comments / reviews -- the bug was
    fundamentally about ordering ("after our last reply, was there a
    push?")."""

    def test_push_event_after_bot_comment_appears_after_it_in_list(self) -> None:
        # Mirrors PR #23197: bot commented at 01:52 about the old head,
        # then the author force-pushed at 21:46.
        timeline = _load_timeline("ffmpeg_pr_23197_timeline.json")
        comments = [
            {
                "user": {"login": "Forgejo_Fairy"},
                "created_at": "2026-05-25T01:52:46Z",
                "updated_at": "2026-05-25T01:52:46Z",
                "body": (
                    "LLM review: I did a deep pass on "
                    f"{PR_23197_INITIAL_SHA[:12]} over ..."
                ),
            },
        ]

        items = fairy.build_llm_discussion(
            reviews=[], comments=comments, review_comments=[],
            timeline=timeline,
        )

        kinds = [(i["kind"], i.get("created_at") or i.get("submitted_at"))
                 for i in items]
        # Expected order: initial push and the review request (same
        # second, 2026-05-21) -> bot comment (2026-05-25 01:52) ->
        # force-push (2026-05-25 21:46).
        self.assertEqual(
            kinds,
            [
                ("push",           PR_23197_INITIAL_PUSH_AT),
                ("review_request", PR_23197_INITIAL_PUSH_AT),
                ("comment",        "2026-05-25T01:52:46Z"),
                ("push",           PR_23197_FORCE_PUSH_AT),
            ],
        )
        # And the LATEST item is the force-push to the new head -- the
        # signal the triager needs to NOT skip with "waiting on author".
        self.assertEqual(items[-1]["head_sha"], PR_23197_NEW_HEAD_SHA)
        self.assertTrue(items[-1]["is_force_push"])

    def test_omitting_timeline_keeps_legacy_behavior(self) -> None:
        # Existing call sites that have not yet been updated must still
        # work; the parameter is optional and absence means "no push
        # events", matching pre-fix behavior.
        comments = [{
            "user": {"login": "alice"},
            "created_at": "2026-05-25T01:52:46Z",
            "body": "hello",
        }]
        items_default = fairy.build_llm_discussion(
            reviews=[], comments=comments, review_comments=[],
        )
        items_explicit_none = fairy.build_llm_discussion(
            reviews=[], comments=comments, review_comments=[], timeline=None,
        )
        items_empty = fairy.build_llm_discussion(
            reviews=[], comments=comments, review_comments=[], timeline=[],
        )
        self.assertEqual(items_default, items_explicit_none)
        self.assertEqual(items_default, items_empty)
        self.assertEqual([i["kind"] for i in items_default], ["comment"])


class ReviewRequestEventsTests(unittest.TestCase):
    """The review_request timeline entry (real capture: FFmpeg #23197,
    michaelni requested a review from kaweno) reaches the discussion."""

    def test_pr_23197_yields_the_review_request(self) -> None:
        timeline = _load_timeline("ffmpeg_pr_23197_timeline.json")
        requests = fairy.review_request_events_from_timeline(timeline)
        self.assertEqual(requests, [{
            "kind": "review_request",
            "author": "michaelni",
            "reviewer": "kaweno",
            "removed": False,
            "created_at": "2026-05-21T21:23:41Z",
        }])

    def test_the_request_reaches_the_built_discussion(self) -> None:
        timeline = _load_timeline("ffmpeg_pr_23197_timeline.json")
        disc = fairy.build_llm_discussion([], [], [], timeline)
        self.assertEqual(
            [(d["author"], d["reviewer"]) for d in disc
             if d["kind"] == "review_request"],
            [("michaelni", "kaweno")])

    def test_a_bare_verdict_is_kept_bare_noise_is_not(self) -> None:
        reviews = [
            {"user": {"login": "alice"}, "state": "APPROVED",
             "submitted_at": "2026-05-22T00:00:00Z", "body": ""},
            {"user": {"login": "bob"}, "state": "COMMENTED",
             "submitted_at": "2026-05-22T01:00:00Z", "body": ""},
            {"user": {"login": "carol"}, "state": "REQUEST_REVIEW",
             "submitted_at": "2026-05-22T02:00:00Z", "body": ""},
        ]
        disc = fairy.build_llm_discussion(reviews, [], [])
        self.assertEqual([(d["author"], d["state"], d["body"]) for d in disc],
                         [("alice", "APPROVED", "")])


if __name__ == "__main__":
    unittest.main()
