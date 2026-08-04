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

fairy.py options: the flag string through to whatever consumes it.

Every namespace here comes from the real ``fairy.parse_args`` on a
shlex-split flag string, so a renamed flag, a moved ``dest`` or a
dropped default fails here instead of quietly reaching production.
The assertions are on the effect the flag has -- the skip reason, the
gcli command line, the prune horizon -- not on the attribute value.
"""

from __future__ import annotations

import argparse
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

NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
OLD = "2026-06-01T00:00:00Z"

HUMAN_COMMENT = {"user": {"login": "dev"}, "created_at": OLD,
                 "updated_at": OLD, "body": "please have a look"}

FAIRY_CI_HEADS_UP = {"user": {"login": "fairy"},
                     "created_at": "2026-06-02T00:00:00Z",
                     "updated_at": "2026-06-02T00:00:00Z",
                     "body": "heads-up: the / build job is red"}

NEWER_HUMAN_COMMENT = {"user": {"login": "dev"},
                       "created_at": "2026-06-03T00:00:00Z",
                       "updated_at": "2026-06-03T00:00:00Z",
                       "body": "thanks, looking"}

FAILING_CI = [{"context": "/ build", "state": "failure",
               "description": "Tests failed in 2m13s",
               "target_url": "/o/r/actions/runs/1/jobs/0",
               "created_at": OLD, "updated_at": OLD}]

TRIAGE_CMD = "--llm-review-cmd 'wrapper --triage-model openai:mini'"


def pr_ns(flags: str = "") -> argparse.Namespace:
    return fairy.parse_args(shlex.split("--owner o --repo r " + flags))


def open_pr(number: int = 1, title: str = "a fix") -> dict:
    return {"number": number, "state": "open", "mergeable": True,
            "title": title, "updated_at": OLD, "head": {"sha": "h1"},
            "user": {"login": "dev"}, "html_url": f"https://forge/pr/{number}"}


class PrepareCase(unittest.TestCase):
    """Shared harness: ``prepare_pr`` on a real parsed namespace with
    the forge mocked out."""

    def prepare(self, flags: str, comments: list | None = None,
                self_login: str | None = None, statuses: list | None = None,
                pr: dict | None = None):
        with mock.patch.object(
                fairy, "get_pr_discussion",
                return_value=([], comments or [HUMAN_COMMENT], [])), \
                mock.patch.object(fairy.gcli_cache, "get",
                                  return_value={"timeline": []}), \
                mock.patch.object(fairy, "list_commit_statuses",
                                  return_value=FAILING_CI if statuses is None
                                  else statuses), \
                mock.patch.object(fairy, "get_auto_merge_info",
                                  return_value="no"), \
                mock.patch.object(fairy, "attach_ci_failure_logs"):
            return fairy.prepare_pr(
                pr_ns(flags), pr or open_pr(), now=NOW, self_login=self_login,
                wip_re=fairy.compile_wip_regex([]),
                cache=gcli_cache.Cache(),
                discussion_cache_max_age=timedelta(hours=1))


class TriageOnCiFailureTests(PrepareCase):
    """--triage-on-ci-failure decides whether a CI-red PR stops at a
    skip or reaches the triage LLM with a failure payload."""

    def test_without_the_flag_red_ci_stops_at_a_skip(self) -> None:
        decision = self.prepare(TRIAGE_CMD + " --patch-repo /p")
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "CI not successful: / build")

    def test_the_flag_sends_the_failing_contexts_to_the_llm(self) -> None:
        prepared = self.prepare(
            "--triage-on-ci-failure " + TRIAGE_CMD + " --patch-repo /p")
        self.assertEqual(
            prepared.ci_triage["contexts_still_requiring_announcement"],
            ["/ build"])
        self.assertIn("CI triage", prepared.base_reason)

    def test_the_flag_needs_an_llm_review_cmd_and_says_so(self) -> None:
        decision = self.prepare("--triage-on-ci-failure")
        self.assertEqual(decision.action, "skip")
        self.assertIn("requires --llm-review-cmd", decision.reason)

    def test_the_flag_needs_a_triage_model_and_says_so(self) -> None:
        decision = self.prepare(
            "--triage-on-ci-failure --llm-review-cmd wrapper --patch-repo /p")
        self.assertEqual(decision.action, "skip")
        self.assertIn("needs --triage-model", decision.reason)

    def test_an_already_announced_failure_still_reaches_the_wrapper(self) -> None:
        """An all-announced red PR flows to the wrapper instead of
        skipping; the empty ``contexts_still_requiring_announcement``
        is what switches the triager off the announce-mode prompt
        (pr_review_wrapper.ci_announce_pending), so a red-CI PR can
        still reach a full review."""
        prepared = self.prepare(
            "--triage-on-ci-failure " + TRIAGE_CMD + " --patch-repo /p",
            comments=[FAIRY_CI_HEADS_UP, NEWER_HUMAN_COMMENT],
            self_login="fairy")
        self.assertEqual(
            prepared.ci_triage["contexts_bot_already_mentioned"], ["/ build"])
        self.assertEqual(
            prepared.ci_triage["contexts_still_requiring_announcement"], [])
        self.assertIn("already announced", prepared.base_reason)

    def test_a_newly_red_job_still_reaches_the_wrapper(self) -> None:
        prepared = self.prepare(
            "--triage-on-ci-failure " + TRIAGE_CMD + " --patch-repo /p",
            comments=[FAIRY_CI_HEADS_UP, NEWER_HUMAN_COMMENT],
            self_login="fairy",
            statuses=FAILING_CI + [dict(FAILING_CI[0], context="/ fate")])
        self.assertEqual(
            prepared.ci_triage["contexts_bot_already_mentioned"], ["/ build"])
        self.assertEqual(
            prepared.ci_triage["contexts_still_requiring_announcement"],
            ["/ fate"])


class MissingCiGateTests(PrepareCase):
    """An empty commit-status list gates only the rule-only auto-approve
    path; an LLM review proceeds without CI evidence.

    Regression: FFmpeg/fateserver runs no CI, so the statuses fetch for
    https://code.ffmpeg.org/FFmpeg/fateserver/pulls/2 (this fixture)
    genuinely returns [] and the PR sat in skipped/ forever with
    "no commit statuses / CI results found"."""

    FATESERVER_PR2 = {
        "number": 2, "state": "open", "mergeable": True,
        "title": "Opinionated list of fixes", "updated_at": OLD,
        "head": {"sha": "4b23f6da3b3b1119958b7f2c0011a1ed7028d5cd"},
        "user": {"login": "carol"},
        "html_url": "https://code.ffmpeg.org/FFmpeg/fateserver/pulls/2"}

    def test_an_llm_review_proceeds_without_any_statuses(self) -> None:
        prepared = self.prepare("--llm-review-cmd wrapper --patch-repo /p",
                                statuses=[], pr=self.FATESERVER_PR2)
        self.assertEqual(prepared.base_reason,
                         "matches all rules; CI contexts=0")

    def test_the_rule_only_auto_approve_still_skips(self) -> None:
        decision = self.prepare("", statuses=[], pr=self.FATESERVER_PR2)
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason,
                         "no commit statuses / CI results found")

    def test_statuses_that_do_exist_still_gate_the_llm_review(self) -> None:
        decision = self.prepare("--llm-review-cmd wrapper --patch-repo /p")
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "CI not successful: / build")


class ApproveMessageTests(unittest.TestCase):
    """--approve-message rides on the gcli approve command as the -T
    body file, combined with the LLM's own message."""

    def approve_body(self, flags: str, review_message: str) -> str | None:
        body: list[str] = []

        def fake_run_cmd(cmd, **kwargs):
            if "-T" in cmd:
                body.append(Path(cmd[cmd.index("-T") + 1]).read_text())
            return argparse.Namespace(returncode=0, stderr="")

        with mock.patch.object(forge_gcli, "run_cmd", side_effect=fake_run_cmd):
            fairy.gcli_approve(pr_ns(flags), 1, review_message)
        return body[0] if body else None

    def test_the_message_is_posted_with_the_approval(self) -> None:
        self.assertEqual(
            self.approve_body("--approve-message 'Thanks, LGTM.'", ""),
            "Thanks, LGTM.")

    def test_it_precedes_the_llm_message(self) -> None:
        self.assertEqual(
            self.approve_body("--approve-message 'Thanks.'", "No issues found."),
            "Thanks.\n\nNo issues found.")

    def test_without_it_a_bare_approval_carries_no_body(self) -> None:
        self.assertIsNone(self.approve_body("", ""))


class ScanCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))

    def scan(self, ns: argparse.Namespace, prs: list[dict],
             prepare=None) -> None:
        with mock.patch.object(fairy, "list_open_prs", return_value=prs), \
                mock.patch.object(fairy, "get_pr",
                                  side_effect=lambda ns, n: open_pr(n)), \
                mock.patch.object(forge_gcli, "self_login", return_value=None), \
                mock.patch.object(agent.gcli_cache, "load_cache",
                                  return_value=gcli_cache.Cache()), \
                mock.patch.object(agent.gcli_cache, "save_cache"), \
                (mock.patch.object(fairy, "safe_prepare_pr", prepare)
                 if prepare is not None else mock.patch.object(
                     fairy, "get_pr_discussion",
                     side_effect=_PastGates(PAST_GATES))):
            agent.scan_pass(self.db, ns, None, now=NOW)


PAST_GATES = "the scan reached the discussion fetch"


class _PastGates(Exception):
    """Sentinel: the scan reached past the title gates into the forge."""


class WipPrefixTests(ScanCase):
    """--wip-prefix extends the title gate the agent compiles for the
    scan; a matching PR never costs a discussion fetch."""

    def test_a_custom_prefix_skips_the_pr_before_any_fetch(self) -> None:
        self.scan(pr_ns("--wip-prefix SPIKE:"), [open_pr(1, "SPIKE: try this")])
        self.assertEqual(self.db.get("skipped", "pr", "1")["reason"],
                         "marked WIP/draft")

    def test_an_unknown_prefix_is_not_a_gate(self) -> None:
        self.scan(pr_ns(), [open_pr(1, "SPIKE: try this")])
        self.assertIn(PAST_GATES, self.db.get("error", "pr", "1")["error"])

    def test_the_builtin_prefixes_survive_a_custom_one(self) -> None:
        self.scan(pr_ns("--wip-prefix SPIKE:"), [open_pr(1, "WIP: try this")])
        self.assertEqual(self.db.get("skipped", "pr", "1")["reason"],
                         "marked WIP/draft")


class CacheDefaultTests(unittest.TestCase):
    """Without --cache the PR side derives its own pickle from the side
    identity, so no two concurrently running sides share a file."""

    def test_the_default_is_derived_from_the_side_identity(self) -> None:
        self.assertEqual(pr_ns().cache,
                         common.default_cache_path("gitea__o_r_pulls.pkl"))

    def test_an_explicit_path_wins(self) -> None:
        self.assertEqual(pr_ns("--cache x.pkl").cache, Path("x.pkl"))


class DiscussionCacheMaxAgeTests(ScanCase):
    """--discussion-cache-max-age-hours is the TTL the scan hands the
    gates for the cached reviews/comments trio."""

    def ttl_for(self, flags: str) -> timedelta:
        prepare = mock.Mock(side_effect=lambda ns, pr, **kw: fairy.Decision(
            pr["number"], "t", "dev", "-", "skip", "no activity", None))
        self.scan(pr_ns(flags), [open_pr(1)], prepare=prepare)
        return prepare.call_args.kwargs["discussion_cache_max_age"]

    def test_the_flag_sets_the_ttl_the_gates_see(self) -> None:
        self.assertEqual(self.ttl_for("--discussion-cache-max-age-hours 6"),
                         timedelta(hours=6))

    def test_the_default_is_a_day(self) -> None:
        self.assertEqual(self.ttl_for(""), timedelta(hours=24))


class WorksetRetentionDaysTests(ScanCase):
    """--workset-retention-days is the age at which the scan prunes the
    settled tickets of items that left the open listing."""

    def settled_survives(self, flags: str, age_days: float) -> bool:
        self.db.push("posted", "pr", "99", {"title": "long merged"})
        data = self.db.get("posted", "pr", "99")
        data["state_changed_at"] = (
            NOW - timedelta(days=age_days)).isoformat()
        self.db._write(self.db.path("posted", "pr", "99"), data)
        self.scan(pr_ns(flags), [])
        return self.db.get("posted", "pr", "99") is not None

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
