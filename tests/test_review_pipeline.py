"""Provider factory + review_pr orchestration, with fake reviewers.

Pins the composable seam: ``make_reviewer`` parses ``provider:model`` into
the right backend, and ``review_pr`` fans the model reviewers out, collects
their drafts on the context, and hands them to the combiner. Real SDK calls
are never made; reviewers are fakes implementing the shared interface.
"""

from __future__ import annotations

import argparse
import sys
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
from llm_review_api import Review, ReviewContext, Reviewer, Z_AI_ANTHROPIC_URL  # noqa: E402
import review_pipeline  # noqa: E402
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
        reasoning_effort="high", service_tier="flex",
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

    def test_effort_suffix_overrides_openai_reasoning_effort(self) -> None:
        r = review_pipeline.make_reviewer("openai:gpt-5.5@xhigh", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertEqual("gpt-5.5", r.model)
        self.assertEqual("xhigh", r.effort)
        # Without a suffix the shared --reasoning-effort applies.
        r = review_pipeline.make_reviewer("openai:gpt-5.5", args=_args(), resources=None, role=REVIEWER_ROLE, verbose=False)
        self.assertEqual("high", r.effort)

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


if __name__ == "__main__":
    unittest.main()
