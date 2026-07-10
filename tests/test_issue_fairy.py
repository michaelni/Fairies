"""Tests for the issue-helper path: verdict schema, issue-vs-PR discovery
filtering, prepare_issue gates, the LLM payload, and issue-label gcli
command construction. Fixtures under fixtures/issue_fairy/ are real
captures from code.ffmpeg.org (FFmpeg/FFmpeg, July 2026)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bot_state  # noqa: E402
import forge_gcli  # noqa: E402
import llm_review_api  # noqa: E402
import issue_fairy  # noqa: E402
from fairy import Decision, LLMReview  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "issue_fairy"


def load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())


def make_args(**overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        owner="FFmpeg",
        repo="FFmpeg",
        gcli_account=None,
        forge_type="gitea",
        verbose=0,
        min_age_days=1.0,
        llm_review_cmd="wrapper-cmd",
        podman_host=None,
        llm_timeout=60,
        llm_max_attempts=1,
        llm_retry_delay=0.0,
        triage_labels=[],
        force_review_issues=set(),
        force_skip_issues=set(),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class IssueReportSchemaTests(unittest.TestCase):
    def test_accepts_every_issue_classification(self) -> None:
        for classification in llm_review_api.ISSUE_REPORT_CLASSIFICATIONS:
            result = llm_review_api.validate_issue_report(
                {"classification": classification, "message": "m"},
            )
            self.assertEqual(result["classification"], classification)

    def test_rejects_pr_classification(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.validate_issue_report(
                {"classification": "approve", "message": "m"},
            )

    def test_sanitizes_labels_against_allowlist(self) -> None:
        result = llm_review_api.validate_result_with_labels(
            {
                "classification": "duplicate",
                "message": "m",
                "label_changes": [
                    {"label": "duplicate", "op": "add", "reason": "dup of #1", "post": True},
                    {"label": "secret", "op": "add", "reason": "", "post": False},
                ],
            },
            ["duplicate"],
            llm_review_api.validate_issue_report,
        )
        self.assertEqual(
            [c["label"] for c in result["label_changes"]], ["duplicate"],
        )


class DiscoveryFilterTests(unittest.TestCase):
    """The forge's /issues listing contains every open PR as an issue;
    discovery must subtract them. Real capture: 20 open items of which
    only 5 are real issues."""

    def test_open_prs_removed_from_listing(self) -> None:
        listing = load_fixture("ffmpeg_issues_listing.json")
        pulls = load_fixture("ffmpeg_open_pulls.json")
        args = make_args()
        with mock.patch.object(issue_fairy, "gcli_api", return_value=listing), \
             mock.patch.object(issue_fairy, "list_open_prs", return_value=pulls):
            issues = issue_fairy.list_open_issues(args)
        self.assertEqual(
            [i["number"] for i in issues],
            [23757, 23749, 23746, 23738, 23737],
        )


def real_issue() -> dict[str, object]:
    listing = load_fixture("ffmpeg_issues_listing.json")
    return next(i for i in listing if i["number"] == 23738)


def prepare(args: argparse.Namespace, issue: dict[str, object], *,
            now: datetime, self_login: str | None = "fairy",
            comments: list | None = None) -> object:
    if comments is None:
        comments = load_fixture("ffmpeg_issue_23738_comments.json")
    timeline = load_fixture("ffmpeg_issue_23738_timeline.json")
    with mock.patch.object(
        issue_fairy, "get_issue_discussion", return_value=(comments, timeline),
    ):
        return issue_fairy.prepare_issue(
            args, issue,
            now=now,
            self_login=self_login,
            cache=mock.Mock(),
            state=bot_state.State(),
            discussion_cache_max_age=timedelta(hours=24),
        )


class PrepareIssueGateTests(unittest.TestCase):
    """Gates exercised with real issue #23738 (8 comments; the newest
    one was created 2026-07-09T13:08:15Z and last edited at 13:15:43Z,
    which is the effective last activity)."""

    LAST_COMMENT = datetime(2026, 7, 9, 13, 15, 43, tzinfo=timezone.utc)

    def test_fresh_activity_skips(self) -> None:
        d = prepare(make_args(), real_issue(), now=self.LAST_COMMENT + timedelta(hours=2))
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "activity is newer than threshold")

    def test_stale_issue_is_prepared(self) -> None:
        p = prepare(make_args(), real_issue(), now=self.LAST_COMMENT + timedelta(days=2))
        self.assertIsInstance(p, issue_fairy.PreparedIssue)
        self.assertEqual(p.number, 23738)
        # The discussion carries the real comments for the LLM.
        self.assertEqual(
            sum(1 for item in p.discussion if item["kind"] == "comment"), 8,
        )

    def test_no_activity_since_own_reply_skips(self) -> None:
        # When the newest comment is fairy's own, there is nothing to
        # react to no matter how stale the issue is.
        comments = load_fixture("ffmpeg_issue_23738_comments.json")
        comments[-1]["user"]["login"] = "fairy"
        d = prepare(
            make_args(), real_issue(),
            now=self.LAST_COMMENT + timedelta(days=30), comments=comments,
        )
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "no activity since fairy's last reply")

    def test_mention_after_own_reply_forces(self) -> None:
        # A fresh @fairy mention overrides both min-age and the
        # already-replied gate.
        comments = load_fixture("ffmpeg_issue_23738_comments.json")
        comments[0]["user"]["login"] = "fairy"
        comments[-1]["body"] += "\n@fairy please bisect this"
        p = prepare(
            make_args(), real_issue(),
            now=self.LAST_COMMENT + timedelta(hours=1), comments=comments,
        )
        self.assertIsInstance(p, issue_fairy.PreparedIssue)
        self.assertEqual(p.base_reason, "later discussion mentions fairy")

    def test_closed_issue_skips_unless_forced(self) -> None:
        issue = dict(real_issue())
        issue["state"] = "closed"
        now = self.LAST_COMMENT + timedelta(days=2)
        d = prepare(make_args(), issue, now=now)
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "not open")
        p = prepare(make_args(force_review_issues={23738}), issue, now=now)
        self.assertIsInstance(p, issue_fairy.PreparedIssue)

    def test_force_skip_wins(self) -> None:
        d = prepare(
            make_args(force_skip_issues={23738}, force_review_issues={23738}),
            real_issue(), now=self.LAST_COMMENT + timedelta(days=2),
        )
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "forced skip by --force-skip-issue")


class LLMPayloadTests(unittest.TestCase):
    def test_payload_shape_and_task_flag(self) -> None:
        args = make_args(triage_labels=["duplicate", "invalid"])
        p = prepare(args, real_issue(),
                    now=PrepareIssueGateTests.LAST_COMMENT + timedelta(days=2))
        self.assertIsInstance(p, issue_fairy.PreparedIssue)
        with mock.patch.object(
            issue_fairy, "invoke_llm_wrapper",
            return_value=LLMReview("needs_info", "m"),
        ) as invoke:
            issue_fairy.run_llm_issue(args, p, None)
        payload = invoke.call_args.args[1]
        kwargs = invoke.call_args.kwargs
        self.assertEqual(payload["issue"]["number"], 23738)
        self.assertEqual(payload["issue"]["author"], "oakaigh")
        self.assertNotIn("pull_request", payload)
        self.assertNotIn("patch", payload)
        self.assertEqual(payload["triage_label_allowlist"], ["duplicate", "invalid"])
        self.assertEqual(kwargs["extra_cmd_args"], ["--task", "issue"])
        self.assertEqual(
            kwargs["allowed_classifications"],
            frozenset(llm_review_api.ISSUE_REPORT_CLASSIFICATIONS),
        )

    def test_non_skip_classification_becomes_comment(self) -> None:
        args = make_args()
        p = prepare(args, real_issue(),
                    now=PrepareIssueGateTests.LAST_COMMENT + timedelta(days=2))
        with mock.patch.object(
            issue_fairy, "run_llm_issue",
            return_value=LLMReview("duplicate", "dup of #1"),
        ):
            d = issue_fairy.evaluate_issue(args, p)
        self.assertEqual(d.action, "comment")
        self.assertEqual(d.llm_classification, "duplicate")
        with mock.patch.object(
            issue_fairy, "run_llm_issue", return_value=LLMReview("skip", ""),
        ):
            d = issue_fairy.evaluate_issue(args, p)
        self.assertEqual(d.action, "skip")


class IssueLabelCommandTests(unittest.TestCase):
    def test_uses_gcli_issues_labels(self) -> None:
        args = make_args()
        with mock.patch("forge_gcli.run_cmd") as run_cmd:
            run_cmd.return_value = subprocess.CompletedProcess([], 0, "", "")
            forge_gcli.apply_issue_label_changes(
                args, "FFmpeg", "FFmpeg", 23738,
                ["duplicate"], ["needs sample"], {"needs sample"},
                kind=forge_gcli.KIND_ISSUE,
            )
        self.assertEqual(
            run_cmd.call_args[0][0],
            [
                "gcli", "-t", "gitea",
                "issues", "-o", "FFmpeg", "-r", "FFmpeg", "-i", "23738",
                "labels", "add", "duplicate", "remove", "needs sample",
            ],
        )


if __name__ == "__main__":
    unittest.main()
