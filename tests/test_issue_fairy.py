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

    def test_rejects_pr_and_disposition_classifications(self) -> None:
        # Dispositions are labels, not classifications; only the two
        # process decisions (reply/skip) are valid verdicts.
        for classification in ("approve", "duplicate", "needs_info"):
            with self.assertRaises(llm_review_api.SchemaError):
                llm_review_api.validate_issue_report(
                    {"classification": classification, "message": "m"},
                )

    def test_sanitizes_labels_against_allowlist(self) -> None:
        result = llm_review_api.validate_result_with_labels(
            {
                "classification": "reply",
                "message": "m",
                "label_changes": [
                    {"label": "resolution/duplicate", "op": "add", "reason": "dup of #1", "post": True},
                    {"label": "secret", "op": "add", "reason": "", "post": False},
                ],
            },
            ["resolution/duplicate"],
            llm_review_api.validate_issue_report,
        )
        self.assertEqual(
            [c["label"] for c in result["label_changes"]], ["resolution/duplicate"],
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
        self.assertEqual(d.reason, "no non-bot activity since fairy's last reply")

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


def with_labels(issue: dict[str, object], *names: str) -> dict[str, object]:
    return {**issue, "labels": [{"name": n} for n in names]}


class LabelGateTests(unittest.TestCase):
    """The forge labels are the analysis state machine: resolution/*
    means done, repro/* marks a completed pass, "needs info" means
    waiting on a human response."""

    STALE = PrepareIssueGateTests.LAST_COMMENT + timedelta(days=30)

    def test_resolution_label_skips(self) -> None:
        d = prepare(make_args(), with_labels(real_issue(), "resolution/wontfix", "bug"),
                    now=self.STALE)
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "resolved: resolution/wontfix")

    def test_repro_label_skips_as_analyzed(self) -> None:
        d = prepare(make_args(), with_labels(real_issue(), "repro/yes", "bug"),
                    now=self.STALE)
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "already analyzed: repro/* set")

    def test_needs_info_waiting_skips(self) -> None:
        # Fairy asked for info (last comment is fairy's); nothing new
        # from humans -> keep waiting, even though repro/no(env) alone
        # would already skip.
        comments = load_fixture("ffmpeg_issue_23738_comments.json")
        comments[-1]["user"]["login"] = "fairy"
        d = prepare(
            make_args(),
            with_labels(real_issue(), "needs info", "repro/no(env)"),
            now=self.STALE, comments=comments,
        )
        self.assertIsInstance(d, Decision)
        self.assertEqual(d.reason, "no non-bot activity since fairy's last reply")

    def test_needs_info_response_reanalyzes(self) -> None:
        # Fairy asked (second-to-last comment), the reporter answered
        # (last comment): the issue is re-analyzed despite repro/no(env)
        # being set, and despite being far younger than --min-age-days
        # (once engaged the threshold drops to hours).
        comments = load_fixture("ffmpeg_issue_23738_comments.json")
        comments[-2]["user"]["login"] = "fairy"
        p = prepare(
            make_args(min_age_days=14.0),
            with_labels(real_issue(), "needs info", "repro/no(env)"),
            now=PrepareIssueGateTests.LAST_COMMENT + timedelta(hours=8),
            comments=comments,
        )
        self.assertIsInstance(p, issue_fairy.PreparedIssue)

    def test_mention_overrides_labels(self) -> None:
        comments = load_fixture("ffmpeg_issue_23738_comments.json")
        comments[-1]["body"] += "\n@fairy is this really not reproducible?"
        p = prepare(
            make_args(),
            with_labels(real_issue(), "repro/no", "resolution/invalid"),
            now=PrepareIssueGateTests.LAST_COMMENT + timedelta(hours=1),
            comments=comments,
        )
        self.assertIsInstance(p, issue_fairy.PreparedIssue)
        self.assertEqual(p.base_reason, "later discussion mentions fairy")


class LLMPayloadTests(unittest.TestCase):
    def test_payload_shape_and_task_flag(self) -> None:
        args = make_args(triage_labels=["duplicate", "invalid"])
        p = prepare(args, real_issue(),
                    now=PrepareIssueGateTests.LAST_COMMENT + timedelta(days=2))
        self.assertIsInstance(p, issue_fairy.PreparedIssue)
        with mock.patch.object(
            issue_fairy, "invoke_llm_wrapper",
            return_value=LLMReview("reply", "m"),
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

    def test_reply_classification_becomes_comment(self) -> None:
        args = make_args()
        p = prepare(args, real_issue(),
                    now=PrepareIssueGateTests.LAST_COMMENT + timedelta(days=2))
        with mock.patch.object(
            issue_fairy, "run_llm_issue",
            return_value=LLMReview("reply", "dup of #1"),
        ):
            d = issue_fairy.evaluate_issue(args, p)
        self.assertEqual(d.action, "comment")
        self.assertEqual(d.llm_classification, "reply")
        with mock.patch.object(
            issue_fairy, "run_llm_issue", return_value=LLMReview("skip", ""),
        ):
            d = issue_fairy.evaluate_issue(args, p)
        self.assertEqual(d.action, "skip")


class PipelineTests(unittest.TestCase):
    """start_issue_pipeline yields exactly one (prepared, decision) per
    issue; candidates beyond --limit are skipped without an LLM call."""

    def _run(self, args: argparse.Namespace, numbers: list[int]) -> list:
        issues = [{"number": n} for n in numbers]

        def fake_prepare(a, issue, **kw):
            return issue_fairy.PreparedIssue(
                issue=issue, number=issue["number"], title="t", author="a",
                last_activity=None, base_reason="stale", discussion=[],
                reviewer_username="fairy",
            )

        def fake_evaluate(a, p):
            return Decision(p.number, p.title, p.author, "-", "comment",
                            "llm", None, "reply", "m")

        with mock.patch.object(issue_fairy, "prepare_issue", side_effect=fake_prepare), \
             mock.patch.object(issue_fairy, "evaluate_issue", side_effect=fake_evaluate), \
             mock.patch.object(issue_fairy, "writeback_llm_skip_backoff"):
            reviewed_queue, llm_queue = issue_fairy.start_issue_pipeline(
                args, issues,
                now=datetime.now(timezone.utc), self_login="fairy",
                cache=mock.Mock(), state=bot_state.State(),
                discussion_cache_max_age=timedelta(hours=1),
            )
            results = [reviewed_queue.get(timeout=10) for _ in issues]
            for _ in range(max(1, args.llm_parallelism)):
                llm_queue.put(issue_fairy._LLM_DONE)
        return results

    def test_parallel_workers_evaluate_every_issue(self) -> None:
        results = self._run(make_args(llm_parallelism=3, limit=0), [1, 2, 3, 4, 5])
        self.assertEqual(sorted(d.pr_number for _, d in results), [1, 2, 3, 4, 5])
        self.assertTrue(all(d.llm_classification == "reply" for _, d in results))

    def test_limit_caps_llm_evaluations(self) -> None:
        results = self._run(make_args(llm_parallelism=2, limit=2), [1, 2, 3, 4, 5])
        evaluated = [d for _, d in results if d.llm_classification == "reply"]
        skipped = [d for _, d in results if "--limit" in d.reason]
        self.assertEqual(len(evaluated), 2)
        self.assertEqual(len(skipped), 3)


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
