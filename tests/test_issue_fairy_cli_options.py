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

issue_fairy.py options: the flag string through to what consumes it.

Namespaces come from the real ``issue_fairy.parse_args`` on a
shlex-split flag string -- ``--issue-label`` in particular lands on
``triage_labels`` after two transforms (per-value CSV split, then a
flatten across repeats), so only a test that starts at the command
line can catch that chain breaking.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent  # noqa: E402
import common  # noqa: E402
import fairy  # noqa: E402
import forge_gcli  # noqa: E402
import filedb  # noqa: E402
import gcli_cache  # noqa: E402
import issue_fairy  # noqa: E402

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
OLD = "2026-06-01T00:00:00Z"


def issue_ns(flags: str = "") -> argparse.Namespace:
    return issue_fairy.parse_args(shlex.split("--owner o --repo r " + flags))


def make_issue(number: int = 5, state: str = "open") -> dict:
    return {"number": number, "title": f"i{number}", "state": state,
            "user": {"login": "dev"}, "body": "it crashes",
            "created_at": OLD, "updated_at": OLD,
            "html_url": f"https://forge/issue/{number}", "labels": []}


def prepared(issue: dict) -> issue_fairy.PreparedIssue:
    return issue_fairy.PreparedIssue(
        issue=issue, number=issue["number"], title=issue["title"],
        author="dev", last_activity=None, base_reason="stale",
        discussion=[], reviewer_username="fairy")


class IssueLabelTests(unittest.TestCase):
    """--issue-label is the LLM's label vocabulary: it rides to the
    wrapper in the payload and bounds what comes back."""

    def run_wrapper(self, flags: str, label_changes: list) -> tuple:
        ns = issue_ns("--llm-review-cmd wrapper " + flags)
        sent: list[str] = []

        def fake_run_cmd(cmd, *, input_text: str, **kwargs):
            sent.append(input_text)
            return argparse.Namespace(returncode=0, stderr="", stdout=json.dumps(
                {"classification": "reply", "message": "m",
                 "label_changes": label_changes}))

        with mock.patch.object(fairy, "run_cmd", side_effect=fake_run_cmd):
            review = issue_fairy.run_llm_issue(ns, prepared(make_issue()), None)
        return json.loads(sent[0]), review

    def test_repeated_and_comma_separated_values_reach_the_wrapper(self) -> None:
        payload, _ = self.run_wrapper(
            "--issue-label duplicate,invalid --issue-label 'needs info'", [])
        self.assertEqual(payload["triage_label_allowlist"],
                         ["duplicate", "invalid", "needs info"])

    def test_without_the_flag_no_allowlist_is_offered(self) -> None:
        payload, _ = self.run_wrapper("", [])
        self.assertNotIn("triage_label_allowlist", payload)

    def test_a_label_outside_the_allowlist_is_dropped(self) -> None:
        _, review = self.run_wrapper(
            "--issue-label duplicate",
            [{"label": "duplicate", "op": "add", "reason": "dupe of #1"},
             {"label": "wontfix", "op": "add", "reason": "invented"}])
        self.assertEqual([c.label for c in review.label_changes], ["duplicate"])

    def test_without_an_allowlist_every_label_is_dropped(self) -> None:
        _, review = self.run_wrapper(
            "", [{"label": "duplicate", "op": "add", "reason": "dupe of #1"}])
        self.assertEqual(review.label_changes, ())


class ForceReviewIssueTests(unittest.TestCase):
    """--force-review-issue overrides the open-state gate, on the way in
    (prepare) and on the way out (the pre-post staleness guard)."""

    def prepare(self, flags: str, issue: dict):
        with mock.patch.object(issue_fairy, "get_issue_discussion",
                               return_value=([], [])):
            return issue_fairy.prepare_issue(
                issue_ns("--llm-review-cmd wrapper " + flags), issue,
                now=NOW, self_login="fairy",
                cache=gcli_cache.Cache(),
                discussion_cache_max_age=timedelta(hours=1))

    def test_a_closed_issue_is_analyzed_when_named(self) -> None:
        result = self.prepare("--force-review-issue 5", make_issue(5, "closed"))
        self.assertEqual(result.base_reason,
                         "forced review by --force-review-issue")

    def test_an_unnamed_closed_issue_is_skipped(self) -> None:
        result = self.prepare("--force-review-issue 6", make_issue(5, "closed"))
        self.assertEqual(result.reason, "not open")

    def test_comma_separated_and_repeated_numbers_all_count(self) -> None:
        flags = "--force-review-issue 5,6 --force-review-issue 7"
        for number in (5, 6, 7):
            with self.subTest(number=number):
                result = self.prepare(flags, make_issue(number, "closed"))
                self.assertEqual(result.base_reason,
                                 "forced review by --force-review-issue")

    def test_force_skip_issue_still_wins(self) -> None:
        result = self.prepare("--force-review-issue 5 --force-skip-issue 5",
                              make_issue(5))
        self.assertEqual(result.reason, "forced skip by --force-skip-issue")

    def test_the_send_guard_lets_a_forced_closed_issue_be_posted(self) -> None:
        decision = fairy.Decision(5, "i5", "dev", "-", "comment", "llm", None,
                                  "reply", "m")
        with mock.patch.object(issue_fairy, "get_issue",
                               return_value=make_issue(5, "closed")):
            blocked = issue_fairy.check_issue_still_unchanged(
                issue_ns("--force-review-issue 5"), decision)
            unnamed = issue_fairy.check_issue_still_unchanged(
                issue_ns(""), decision)
        self.assertIsNone(blocked)
        self.assertEqual(unnamed, "issue is no longer open")


class IssueScanCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))

    def scan(self, ns: argparse.Namespace, issues: list[dict], prepare=None):
        prepare = prepare or mock.Mock(
            side_effect=lambda ns, issue, **kw: prepared(issue))
        with mock.patch.object(issue_fairy, "list_open_issues",
                               return_value=issues), \
                mock.patch.object(issue_fairy, "prepare_issue", prepare), \
                mock.patch.object(forge_gcli, "self_login", return_value="fairy"), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=gcli_cache.Cache()), \
                mock.patch.object(agent.gcli_cache, "save_cache"):
            agent.scan_pass(self.db, None, ns, now=NOW)
        return prepare


class IssueCacheDefaultTests(unittest.TestCase):
    """Without --cache the issue side derives its own pickle, distinct
    from the PR side's, so the two sides of one repo running
    concurrently never discard each other's whole-file saves."""

    def test_the_default_is_derived_from_the_side_identity(self) -> None:
        self.assertEqual(issue_ns().cache,
                         common.default_cache_path("gitea__o_r_issues.pkl"))

    def test_it_differs_from_the_pr_side_of_the_same_repo(self) -> None:
        self.assertNotEqual(
            issue_ns().cache,
            fairy.parse_args(shlex.split("--owner o --repo r")).cache)

    def test_an_explicit_path_wins(self) -> None:
        self.assertEqual(issue_ns("--cache x.pkl").cache, Path("x.pkl"))


class IssueDiscussionCacheMaxAgeTests(IssueScanCase):
    """--discussion-cache-max-age-hours is the TTL the issue scan hands
    the gates for the cached comments."""

    def ttl_for(self, flags: str) -> timedelta:
        prepare = self.scan(issue_ns(flags), [make_issue(5)])
        return prepare.call_args.kwargs["discussion_cache_max_age"]

    def test_the_flag_sets_the_ttl_the_gates_see(self) -> None:
        self.assertEqual(self.ttl_for("--discussion-cache-max-age-hours 2"),
                         timedelta(hours=2))

    def test_the_default_is_a_day(self) -> None:
        self.assertEqual(self.ttl_for(""), timedelta(hours=24))


class IssueWorksetRetentionDaysTests(IssueScanCase):
    """On an issue-only agent the issue side owns the prune horizon."""

    def settled_survives(self, flags: str, age_days: float) -> bool:
        self.db.push("posted", "issue", "99", {"title": "long closed"})
        data = self.db.get("posted", "issue", "99")
        data["state_changed_at"] = (NOW - timedelta(days=age_days)).isoformat()
        self.db._write(self.db.path("posted", "issue", "99"), data)
        self.scan(issue_ns(flags), [])
        return self.db.get("posted", "issue", "99") is not None

    def test_a_ticket_older_than_the_retention_is_pruned(self) -> None:
        self.assertFalse(self.settled_survives(
            "--workset-retention-days 10", age_days=20))

    def test_a_longer_retention_keeps_the_same_ticket(self) -> None:
        self.assertTrue(self.settled_survives(
            "--workset-retention-days 30", age_days=20))

    def test_the_default_keeps_two_weeks(self) -> None:
        self.assertTrue(self.settled_survives("", age_days=13))
        self.assertFalse(self.settled_survives("", age_days=15))


if __name__ == "__main__":
    unittest.main()
