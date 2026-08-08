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

Provider factory + review_pr orchestration, with fake reviewers.

Pins the composable seam: ``make_reviewer`` parses ``provider:model`` into
the right backend, and ``review_pr`` fans the model reviewers out, collects
their drafts on the context, and hands them to the combiner. Real SDK calls
are never made; reviewers are fakes implementing the shared interface.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# A fake ``anthropic`` so make_reviewer can build Anthropic/GLM reviewers
# without the SDK installed.
if "anthropic" not in sys.modules:
    fake = types.ModuleType("anthropic")

    class _E(Exception):
        pass

    fake.Anthropic = object
    for _name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError"):
        setattr(fake, _name, type(_name, (_E,), {}))
    sys.modules["anthropic"] = fake

from llm_prompt import COMBINER_ROLE, REVIEWER_ROLE  # noqa: E402
from llm_review_api import (Review, ReviewContext, Reviewer,  # noqa: E402
                            ProviderTurnFailed, Z_AI_ANTHROPIC_URL)
import pr_review_wrapper  # noqa: E402
import review_pipeline  # noqa: E402
import workset  # noqa: E402
from openai_reviewer import OpenAIReviewer  # noqa: E402
from anthropic_reviewer import AnthropicReviewer  # noqa: E402


def _ctx() -> ReviewContext:
    return ReviewContext(
        request={}, patch_text="", patch_truncated=False, source_bundle=None,
        source_files=[], source_notes=[], reviewer_username="fairy",
        ci_triage_mode=False, repo_roots=[Path.cwd()], repo_mount_paths=[],
    )


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="gpt-5.4-mini", podman_max_tool_rounds=0, podman_exec_timeout=600.0,
        service_tier="flex", reasoning_summary="auto", verbosity="high",
    )


class _FakeReviewer(Reviewer):
    def __init__(self, name: str, review: Review) -> None:
        self.name = name
        self._review = review
        self.seen_drafts: list[Review] | None = None

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        raise NotImplementedError

    def review(self, ctx: ReviewContext) -> Review:
        self.seen_drafts = list(ctx.review_drafts())
        return self._review


class _FailingReviewer(Reviewer):
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        raise RuntimeError(f"{self.name}: simulated provider failure")


class _FlaggedReviewer(Reviewer):
    """ProviderTurnFailed on the first ``fail_times`` calls, then a draft."""

    def __init__(self, name: str, review: Review, fail_times: int) -> None:
        self.name = name
        self._review = review
        self.fail_times = fail_times
        self.calls = 0

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        raise NotImplementedError

    def review(self, ctx: ReviewContext) -> Review:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderTurnFailed(f"{self.name}: content flagged")
        return self._review


class MakeReviewerTests(unittest.TestCase):
    def test_openai_prefix_builds_openai(self) -> None:
        r = review_pipeline.make_reviewer("openai:gpt-5.4", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertIsInstance(r, OpenAIReviewer)
        self.assertEqual("openai:gpt-5.4", r.name)
        self.assertEqual("gpt-5.4", r.model)

    def test_bare_model_name_rejected(self) -> None:
        # No default provider: openai needs its prefix like everyone else.
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer("gpt-5.4", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)

    def test_zai_uses_anthropic_endpoint(self) -> None:
        r = review_pipeline.make_reviewer("zai:glm-4.6", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertIsInstance(r, AnthropicReviewer)
        self.assertEqual("zai:glm-4.6", r.name)
        self.assertEqual(Z_AI_ANTHROPIC_URL, r.base_url)
        self.assertEqual("ZAI_API_KEY", r.api_key_env)

    def test_anthropic_provider(self) -> None:
        r = review_pipeline.make_reviewer("anthropic:claude-opus-4", args=_args(), resources=None, role=COMBINER_ROLE, verbose=False)
        self.assertIsInstance(r, AnthropicReviewer)
        self.assertEqual("anthropic:claude-opus-4", r.name)
        self.assertIsNone(r.base_url)
        self.assertIs(COMBINER_ROLE, r.role)

    def test_unknown_provider_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer("grok:x", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)

    def test_effort_suffix_and_default_effort(self) -> None:
        r = review_pipeline.make_reviewer("openai:gpt-5.5@xhigh", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False, default_effort="high")
        self.assertEqual("gpt-5.5", r.model)
        self.assertEqual("xhigh", r.effort)
        r = review_pipeline.make_reviewer("openai:gpt-5.5", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False, default_effort="high")
        self.assertEqual("high", r.effort)
        r = review_pipeline.make_reviewer("openai:gpt-5.5", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertIsNone(r.effort)

    def test_effort_suffix_sets_anthropic_thinking_effort(self) -> None:
        r = review_pipeline.make_reviewer("zai:glm-5.2@low", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertIsInstance(r, AnthropicReviewer)
        self.assertEqual("glm-5.2", r.model)
        self.assertEqual("low", r.effort)
        self.assertIsNone(
            review_pipeline.make_reviewer("zai:glm-5.2", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False).effort
        )

    def test_invalid_anthropic_effort_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer("zai:glm-5.2@turbo", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)

    def test_explicit_none_service_tier_is_not_inherited(self) -> None:
        # Regression: the triager passes --triage-service-tier verbatim,
        # documented as independent of --service-tier. An explicit None
        # must send no tier even when --service-tier is set (flex here);
        # only leaving the parameter unset inherits it.
        r = review_pipeline.make_reviewer("openai:gpt-5.4-mini", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False, service_tier=None)
        self.assertIsNone(r.service_tier)
        r = review_pipeline.make_reviewer("openai:gpt-5.4-mini", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertEqual("flex", r.service_tier)


class ReviewPrTests(unittest.TestCase):
    def test_single_reviewer_no_combiner_returns_draft(self) -> None:
        ctx = _ctx()
        draft = Review("minor_issues_approve", "nit", model="a")
        out = review_pipeline.review_pr(ctx, [_FakeReviewer("a", draft)], None)
        self.assertIs(out, draft)
        self.assertEqual([draft], ctx.drafts)

    def test_multiple_reviewers_require_combiner(self) -> None:
        ctx = _ctx()
        with self.assertRaises(SystemExit):
            review_pipeline.review_pr(
                ctx,
                [_FakeReviewer("a", Review("approve", model="a")),
                 _FakeReviewer("b", Review("approve", model="b"))],
                None,
            )

    def test_combiner_sees_all_drafts(self) -> None:
        ctx = _ctx()
        d1 = Review("moderate_issues", "issue1", model="a")
        d2 = Review("major_issues", "issue2", model="b")
        merged = Review("major_issues", "verified+merged", model="combiner")
        combiner = _FakeReviewer("combiner", merged)
        out = review_pipeline.review_pr(ctx, [_FakeReviewer("a", d1), _FakeReviewer("b", d2)], combiner)
        self.assertIs(out, merged)
        self.assertEqual([d1, d2], ctx.drafts)
        self.assertEqual([d1, d2], combiner.seen_drafts)

    def test_on_drafts_reports_before_the_combiner(self) -> None:
        ctx = _ctx()
        d1 = Review("moderate_issues", "issue1", model="a")
        merged = Review("major_issues", "merged", model="c")
        seen: list[list[Review]] = []
        out = review_pipeline.review_pr(
            ctx, [_FakeReviewer("a", d1)], _FakeReviewer("c", merged),
            on_drafts=seen.append,
        )
        self.assertEqual(seen, [[d1]])
        self.assertIs(out, merged)

    def test_failed_reviewer_does_not_discard_surviving_draft(self) -> None:
        # Regression: a z.ai quota exhaustion (RateLimitError 1308) used to
        # abort the whole ensemble review; the GPT draft must survive and
        # still be verified by the combine stage.
        ctx = _ctx()
        survivor = Review("major_issues", "found it", model="openai:gpt-5.4")
        merged = Review("major_issues", "verified", model="combiner")
        combiner = _FakeReviewer("combiner", merged)
        with self.assertLogs("llm_review_api", level="ERROR"):
            out = review_pipeline.review_pr(
                ctx,
                [_FakeReviewer("a", survivor), _FailingReviewer("zai:glm-5.2")],
                combiner,
            )
        self.assertIs(out, merged)
        self.assertEqual([survivor], ctx.drafts)
        self.assertEqual([survivor], combiner.seen_drafts)
        self.assertEqual(
            ["zai:glm-5.2: zai:glm-5.2: simulated provider failure"],
            ctx.failed_reviewers)

    def test_a_provider_ended_turn_is_retried_and_recovers(self) -> None:
        """gpt-5.6-sol's "possible cybersecurity risk" flag on PR #23901:
        transient, so a retry -- of that reviewer alone -- recovers the
        full ensemble."""
        ctx = _ctx()
        d1 = Review("minor_issues_approve", "ok", model="glm")
        d2 = Review("major_issues", "found it", model="gpt")
        flagged = _FlaggedReviewer("codex:gpt", d2, fail_times=1)
        merged = Review("major_issues", "verified", model="combiner")
        with self.assertLogs("llm_review_api", level="WARNING"):
            out = review_pipeline.review_pr(
                ctx, [_FakeReviewer("zai:glm", d1), flagged],
                _FakeReviewer("combiner", merged))
        self.assertIs(out, merged)
        self.assertEqual([d1, d2], ctx.drafts)
        self.assertEqual(2, flagged.calls)
        self.assertEqual([], ctx.failed_reviewers)

    def test_a_flagged_combiner_is_retried_to_its_budget(self) -> None:
        ctx = _ctx()
        d1 = Review("moderate_issues", "issue", model="glm")
        merged = Review("moderate_issues", "verified", model="combiner")
        combiner = _FlaggedReviewer("codex:gpt", merged, fail_times=7)
        with self.assertLogs("llm_review_api", level="WARNING"):
            out = review_pipeline.review_pr(
                ctx, [_FakeReviewer("zai:glm", d1)], combiner)
        self.assertIs(out, merged)
        self.assertEqual(8, combiner.calls)

    def test_an_exhausted_combiner_propagates(self) -> None:
        """The error state's doubling backoff owns the persistent case:
        the wrapper maps this to EXIT_TURN_FAILED, which also skips the
        outer --llm-max-attempts loop."""
        ctx = _ctx()
        d1 = Review("moderate_issues", "issue", model="glm")
        combiner = _FlaggedReviewer("codex:gpt", d1, fail_times=8)
        with self.assertRaises(ProviderTurnFailed), \
                self.assertLogs("llm_review_api", level="WARNING"):
            review_pipeline.review_pr(
                ctx, [_FakeReviewer("zai:glm", d1)], combiner)
        self.assertEqual(8, combiner.calls)

    def test_an_exhausted_reviewer_is_dropped_and_named(self) -> None:
        ctx = _ctx()
        d1 = Review("minor_issues_approve", "ok", model="glm")
        flagged = _FlaggedReviewer("codex:gpt", d1, fail_times=4)
        merged = Review("minor_issues_approve", "verified", model="combiner")
        with self.assertLogs("llm_review_api", level="ERROR"):
            out = review_pipeline.review_pr(
                ctx, [_FakeReviewer("zai:glm", d1), flagged],
                _FakeReviewer("combiner", merged))
        self.assertIs(out, merged)
        self.assertEqual(4, flagged.calls)
        self.assertEqual(["codex:gpt: codex:gpt: content flagged"],
                         ctx.failed_reviewers)

    def test_a_flagged_single_reviewer_is_retried(self) -> None:
        ctx = _ctx()
        d1 = Review("minor_issues_approve", "ok", model="gpt")
        flagged = _FlaggedReviewer("codex:gpt", d1, fail_times=1)
        merged = Review("minor_issues_approve", "verified", model="combiner")
        with self.assertLogs("llm_review_api", level="WARNING"):
            out = review_pipeline.review_pr(
                ctx, [flagged], _FakeReviewer("combiner", merged))
        self.assertIs(out, merged)
        self.assertEqual(2, flagged.calls)

    def test_single_reviewer_with_combiner_still_combines(self) -> None:
        ctx = _ctx()
        draft = Review("moderate_issues", "issue", model="a")
        merged = Review("minor_issues_approve", "verified", model="combiner")
        combiner = _FakeReviewer("combiner", merged)
        out = review_pipeline.review_pr(ctx, [_FakeReviewer("a", draft)], combiner)
        self.assertIs(out, merged)
        self.assertEqual([draft], combiner.seen_drafts)

    def test_combiner_still_merges_when_two_of_three_survive(self) -> None:
        ctx = _ctx()
        d1 = Review("minor_issues_approve", "nit", model="a")
        d3 = Review("major_issues", "bug", model="c")
        merged = Review("major_issues", "merged", model="combiner")
        combiner = _FakeReviewer("combiner", merged)
        with self.assertLogs("llm_review_api", level="ERROR"):
            out = review_pipeline.review_pr(
                ctx,
                [_FakeReviewer("a", d1), _FailingReviewer("b"), _FakeReviewer("c", d3)],
                combiner,
            )
        self.assertIs(out, merged)
        self.assertEqual([d1, d3], combiner.seen_drafts)

    def test_all_reviewers_failing_aborts(self) -> None:
        ctx = _ctx()
        with self.assertLogs("llm_review_api", level="ERROR"), self.assertRaises(RuntimeError):
            review_pipeline.review_pr(
                ctx, [_FailingReviewer("a"), _FailingReviewer("b")], None,
            )


class RunTriageTests(unittest.TestCase):
    class _Triager(Reviewer):
        def __init__(self, result: dict[str, object] | None = None, *, fail: bool = False) -> None:
            self.name = "fake:triager"
            self.role = REVIEWER_ROLE
            self._result = result or {"route": "engage", "message": "", "reason": "ok"}
            self._fail = fail

        def run(self, ctx: ReviewContext) -> dict[str, object]:
            if self._fail:
                raise RuntimeError("simulated triage failure")
            return dict(self._result)

    def test_returns_validated_dict(self) -> None:
        result = review_pipeline.run_triage(
            self._Triager({"route": "skip", "message": "", "reason": "waiting"}),
            _ctx(),
        )
        self.assertEqual("skip", result["route"])

    def test_failure_returns_none(self) -> None:
        self.assertIsNone(review_pipeline.run_triage(self._Triager(fail=True), _ctx()))


class WrapperWorksetNoteTests(unittest.TestCase):
    """The wrapper records stage progress and outputs in the caller's file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "pr-9.json"
        import json
        self.path.write_text(json.dumps({"title": "t", "author": "a"}),
                             encoding="utf-8")
        self.args = argparse.Namespace(workset_file=self.path)

    def _raw(self) -> dict:
        import json
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_stage_and_triage_result_recorded(self) -> None:
        pr_review_wrapper.workset_note_stage(self.args, "triage")
        pr_review_wrapper.workset_note_triage(
            self.args, {"route": "engage", "reason": "r"})
        data = self._raw()
        self.assertEqual(data["stage"], "triage")
        self.assertEqual(data["triage"], {"route": "engage", "reason": "r"})
        self.assertEqual(data["title"], "t")  # ticket fields undisturbed

    def test_drafts_recorded_and_combine_stage_entered(self) -> None:
        drafts = [Review(
            "moderate_issues", "d1",
            ({"label": "l", "op": "add", "reason": "", "post": False},),
            model="a",
        )]
        pr_review_wrapper.workset_note_drafts(self.args, drafts, combining=True)
        data = self._raw()
        self.assertEqual(data["stage"], "combine")
        self.assertEqual(data["drafts"][0]["message"], "d1")
        self.assertEqual(data["drafts"][0]["model"], "a")
        self.assertEqual(data["drafts"][0]["label_changes"][0]["label"], "l")

    def test_failed_reviewers_land_on_the_ticket(self) -> None:
        pr_review_wrapper.workset_note_drafts(
            self.args, [], combining=True,
            failed=["codex:gpt-5.6-sol: content flagged"])
        self.assertEqual(self._raw()["failed_reviewers"],
                         ["codex:gpt-5.6-sol: content flagged"])

    def test_notes_work_on_a_stateless_filedb_ticket(self) -> None:
        self.path.write_text('{"title": "t"}\n', encoding="utf-8")
        pr_review_wrapper.workset_note_stage(self.args, "review")
        self.assertEqual(self._raw()["stage"], "review")

    def test_no_workset_file_is_a_noop(self) -> None:
        pr_review_wrapper.workset_note_stage(
            argparse.Namespace(workset_file=None), "triage")


if __name__ == "__main__":
    unittest.main()
