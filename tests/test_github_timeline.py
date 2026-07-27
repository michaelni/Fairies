"""GitHub's timeline reaches fairy as the events she already reads.

Fixtures are unmodified ``gcli -t github api`` captures from the
project's own scratch repository:

* ``testrepo_pr2_forcepush_timeline.json`` -- a mixed feed covering the
  three ways GitHub names an actor (``user`` on a comment or review,
  ``actor`` on a label change) and the several ways it dates an entry,
  plus a real force-push performed against that PR.
* ``testrepo_pr4_timeline.json`` -- a run of ``committed`` entries, the
  case the push grouping exists for, and the ``author`` actor key that
  only a commit entry carries.

GitHub has no equivalent of Forgejo's single ``pull_push`` entry: it
lists each commit and never says where one push ended, so a run of them
is read as one. What it does not say is whether the branch was
force-pushed, so that flag is left absent rather than guessed.
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

    def test_force_push_is_left_unknown_not_claimed_false(self) -> None:
        push = [e for e in _github("testrepo_pr4_timeline.json")
                if e["type"] == forge_gcli.PUSH_EVENT][0]
        self.assertNotIn("is_force_push", push)
        self.assertIsNone(
            fairy.push_events_from_timeline([push])[0]["is_force_push"])

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


if __name__ == "__main__":
    unittest.main()
