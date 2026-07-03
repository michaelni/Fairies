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

from llm_review_api import Review, ReviewContext, Reviewer, Z_AI_ANTHROPIC_URL  # noqa: E402
import openai_pr_review_wrapper as wrapper  # noqa: E402
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
    )


class _FakeReviewer(Reviewer):
    def __init__(self, name: str, review: Review) -> None:
        self.name = name
        self._review = review
        self.seen_drafts: list[Review] | None = None

    def review(self, ctx: ReviewContext) -> Review:
        self.seen_drafts = list(ctx.review_drafts())
        return self._review


class MakeReviewerTests(unittest.TestCase):
    def test_bare_and_openai_prefix_build_openai(self) -> None:
        for spec in ("gpt-5.4", "openai:gpt-5.4"):
            r = wrapper.make_reviewer(spec, args=_args(), resources=None, role="reviewer", verbose=False)
            self.assertIsInstance(r, wrapper.OpenAIReviewer)
            self.assertEqual("openai:gpt-5.4", r.name)
            self.assertEqual("gpt-5.4", r.model)

    def test_zai_uses_anthropic_endpoint(self) -> None:
        r = wrapper.make_reviewer("zai:glm-4.6", args=_args(), resources=None, role="reviewer", verbose=False)
        self.assertIsInstance(r, AnthropicReviewer)
        self.assertEqual("zai:glm-4.6", r.name)
        self.assertEqual(Z_AI_ANTHROPIC_URL, r.base_url)
        self.assertEqual("ZAI_API_KEY", r.api_key_env)

    def test_anthropic_provider(self) -> None:
        r = wrapper.make_reviewer("anthropic:claude-opus-4", args=_args(), resources=None, role="combiner", verbose=False)
        self.assertIsInstance(r, AnthropicReviewer)
        self.assertEqual("anthropic:claude-opus-4", r.name)
        self.assertIsNone(r.base_url)
        self.assertEqual("combiner", r.role)

    def test_unknown_provider_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            wrapper.make_reviewer("grok:x", args=_args(), resources=None, role="reviewer", verbose=False)


class ReviewPrTests(unittest.TestCase):
    def test_single_reviewer_no_combiner_returns_draft(self) -> None:
        ctx = _ctx()
        draft = Review("minor_issues_approve", "nit", model="a")
        out = wrapper.review_pr(ctx, [_FakeReviewer("a", draft)], None)
        self.assertIs(out, draft)
        self.assertEqual([draft], ctx.drafts)

    def test_multiple_reviewers_require_combiner(self) -> None:
        ctx = _ctx()
        with self.assertRaises(SystemExit):
            wrapper.review_pr(
                ctx,
                [_FakeReviewer("a", Review("ok_approve", model="a")),
                 _FakeReviewer("b", Review("ok_approve", model="b"))],
                None,
            )

    def test_combiner_sees_all_drafts(self) -> None:
        ctx = _ctx()
        d1 = Review("moderate_issues_comment", "issue1", model="a")
        d2 = Review("major_request_changes", "issue2", model="b")
        merged = Review("major_request_changes", "verified+merged", model="combiner")
        combiner = _FakeReviewer("combiner", merged)
        out = wrapper.review_pr(ctx, [_FakeReviewer("a", d1), _FakeReviewer("b", d2)], combiner)
        self.assertIs(out, merged)
        self.assertEqual([d1, d2], ctx.drafts)
        self.assertEqual([d1, d2], combiner.seen_drafts)


if __name__ == "__main__":
    unittest.main()
