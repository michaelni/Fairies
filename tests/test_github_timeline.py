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

GitHub's timeline reaches fairy as the events she already reads.

Fixtures are unmodified ``gcli -t github api`` captures from the
project's own scratch repository:

* ``testrepo_pr2_forcepush_timeline.json`` -- a mixed feed covering the
  three ways GitHub names an actor (``user`` on a comment or review,
  ``actor`` on a label change) and the several ways it dates an entry,
  plus a real force-push performed against that PR.
* ``testrepo_pr4_timeline.json`` -- a run of ``committed`` entries, the
  case the push grouping exists for, and the ``author`` actor key that
  only a commit entry carries.

GitHub has no equivalent of Forgejo's single ``pull_push`` entry for an
ordinary push: it lists each commit and never says where one push ended,
so a run of them is read as one. A force-push it does mark, with a
``head_ref_force_pushed`` entry closing the run, and in the capture that
marker's ``commit_id`` is the same sha as the ``committed`` entry before
it. The marker also dates the push -- the commit is stamped when it was
written, which on a rewritten branch is the older time.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import forge_gcli  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "github"


def _timeline(forge_type: str, raw: list[dict]) -> list[dict]:
    args = SimpleNamespace(forge_type=forge_type, gcli_account="", verbose=0)
    with mock.patch.object(forge_gcli, "gcli_api", lambda *a, **k: raw):
        return forge_gcli.list_issue_timeline(args, "o", "r", 1)


def _load(name: str) -> list[dict]:
    with (FIXTURES / name).open() as f:
        return json.load(f)


def _github(name: str) -> list[dict]:
    return _timeline("github", _load(name))


class EventProjectionTests(unittest.TestCase):

    def test_every_event_is_dated(self) -> None:
        # Ordering, the activity gate and --simulate-past all key on
        # created_at; GitHub supplies it under three different names.
        events = _github("testrepo_pr2_forcepush_timeline.json")
        undated = [e["type"] for e in events if not e["created_at"]]
        self.assertEqual(undated, [])

    def test_actor_is_found_under_each_of_githubs_three_keys(self) -> None:
        by_type = {e["type"]: e for e in
                   _github("testrepo_pr2_forcepush_timeline.json")}
        self.assertEqual(by_type["commented"]["user"]["login"],
                         "forgejo-fairy[bot]")
        self.assertEqual(by_type["labeled"]["user"]["login"],
                         "forgejo-fairy[bot]")
        # A commit carries a git identity, which has a name but no login.
        push = [e for e in _github("testrepo_pr4_timeline.json")
                if e["type"] == forge_gcli.PUSH_EVENT][0]
        self.assertIsNone(push["user"]["login"])
        self.assertTrue(push["user"]["full_name"])

    def test_event_carries_the_keys_the_contract_promises(self) -> None:
        for event in _github("testrepo_pr2_forcepush_timeline.json"):
            self.assertLessEqual(
                {"type", "id", "user", "created_at", "body"}, set(event),
            )


class PushGroupingTests(unittest.TestCase):

    def test_a_run_of_commits_becomes_one_push(self) -> None:
        pushes = [e for e in _github("testrepo_pr4_timeline.json")
                  if e["type"] == forge_gcli.PUSH_EVENT]
        self.assertEqual(len(pushes), 1)
        self.assertEqual(len(pushes[0]["commit_ids"]), 7)

    def test_an_unmarked_run_of_commits_is_an_ordinary_push(self) -> None:
        push = [e for e in _github("testrepo_pr4_timeline.json")
                if e["type"] == forge_gcli.PUSH_EVENT][0]
        self.assertFalse(push["is_force_push"])

    def test_a_marked_run_is_a_force_push(self) -> None:
        pushes = [e for e in _github("testrepo_pr2_forcepush_timeline.json")
                  if e["type"] == forge_gcli.PUSH_EVENT]
        self.assertEqual(len(pushes), 1)
        self.assertTrue(pushes[0]["is_force_push"])
        self.assertEqual(
            pushes[0]["commit_ids"],
            ["ef072708722966f8997bb909d09b2b2c0e21d89a"])

    def test_the_push_is_dated_and_attributed_by_the_marker(self) -> None:
        # The commit is stamped 19:12:10, the force-push happened at
        # 19:39:47; the activity gate must see the later one, and the
        # pusher rather than the commit's author.
        item = fairy.push_events_from_timeline(
            _github("testrepo_pr2_forcepush_timeline.json"))[0]
        self.assertEqual(item["created_at"], "2026-07-28T19:39:47Z")
        self.assertEqual(item["author"], "michaelni")
        self.assertTrue(item["is_force_push"])

    def test_the_push_reaches_fairy_as_a_discussion_item(self) -> None:
        items = fairy.push_events_from_timeline(
            _github("testrepo_pr4_timeline.json"))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "push")
        self.assertEqual(items[0]["commit_count"], 7)
        self.assertTrue(items[0]["head_sha"])


class ForgejoTimelineIsUnaffectedTests(unittest.TestCase):
    """The Forgejo shape still takes its own path (both shapes pinned)."""

    def test_forgejo_push_keeps_its_decoded_flag(self) -> None:
        raw = _load("../forgejo_pr_timeline/ffmpeg_pr_23197_timeline.json")
        pushes = [e for e in _timeline("gitea", raw)
                  if e["type"] == forge_gcli.PUSH_EVENT]
        self.assertEqual([p["is_force_push"] for p in pushes], [False, True])

    def test_forgejo_commit_entries_are_not_regrouped(self) -> None:
        # Forgejo never lists bare commits, so the GitHub grouping must
        # not run for it: the feed keeps its own event types.
        raw = _load("../forgejo_pr_timeline/ffmpeg_pr_23197_timeline.json")
        types = {e["type"] for e in _timeline("gitea", raw)}
        self.assertIn("pull_scheduled_merge", types)
        self.assertNotIn("committed", types)


class ReviewRequestProjectionTests(unittest.TestCase):
    """GitHub review requests fold into the Forgejo review_request
    shape. Real capture: testrepo PR #5, where the fairy app requested
    a review from michaelni and withdrew it again
    (``review_requested`` / ``review_request_removed`` with ``actor``
    and ``requested_reviewer``,
    https://docs.github.com/en/rest/issues/timeline). A team request
    carries ``requested_team`` instead and has no capture; the
    synthetic event below follows the documented schema."""

    def test_captured_events_fold_into_the_forgejo_shape(self) -> None:
        got = _github("testrepo_review_request_timeline.json")
        self.assertEqual(
            [(e["type"], (e.get("assignee") or {}).get("login"),
              e["removed_assignee"], e["user"]["login"])
             for e in got if e["type"] == "review_request"],
            [("review_request", "michaelni", False, "forgejo-fairy[bot]"),
             ("review_request", "michaelni", True, "forgejo-fairy[bot]")])

    def test_the_requests_reach_fairy_as_discussion_items(self) -> None:
        items = fairy.review_request_events_from_timeline(
            _github("testrepo_review_request_timeline.json"))
        self.assertEqual(
            [(i["author"], i["reviewer"], i["removed"]) for i in items],
            [("forgejo-fairy[bot]", "michaelni", False),
             ("forgejo-fairy[bot]", "michaelni", True)])

    def test_a_team_request_projects_to_no_assignee(self) -> None:
        got = _timeline("github", [
            {"event": "review_requested", "id": 3,
             "actor": {"login": "michaelni"},
             "requested_team": {"name": "reviewers"},
             "created_at": "2026-07-28T12:00:00Z"}])
        self.assertEqual(
            [(e["type"], e.get("assignee"), e["removed_assignee"])
             for e in got],
            [("review_request", None, False)])


if __name__ == "__main__":
    unittest.main()
