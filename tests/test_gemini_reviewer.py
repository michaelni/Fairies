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

GeminiReviewer on the shared tool loop, replayed with a scripted client.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import fake_genai  # noqa: E402

fake_genai.install()

import gemini_reviewer  # noqa: E402
from gemini_reviewer import types  # noqa: E402
from llm_review_api import ProviderTurnFailed  # noqa: E402
from tests.test_anthropic_reviewer import _ctx, _FakeShell  # noqa: E402

SUBMIT = {"classification": "minor_issues_approve", "message": "LLM review: one nit.",
          "head_vs_branch_diff_evidence": False}


def _reply(*parts: object) -> object:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)), finish_reason="STOP")])


def _call(name: str, args: dict, call_id: str) -> object:
    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=call_id),
                      thought_signature=b"sig-" + call_id.encode())


class _ScriptedClient:
    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _reviewer(client: _ScriptedClient, **kwargs: object) -> gemini_reviewer.GeminiReviewer:
    reviewer = gemini_reviewer.GeminiReviewer(
        "gemini-3.8-flash", name="gemini:gemini-3.8-flash", **kwargs)
    reviewer._client = lambda: client  # type: ignore[method-assign]
    return reviewer


class GeminiReviewLoopTests(unittest.TestCase):
    def test_shell_round_then_submit(self) -> None:
        shell = _FakeShell()
        shell_turn = _reply(_call("shell", {"command": "git log -1"}, "c1"))
        client = _ScriptedClient([shell_turn, _reply(_call("submit_review", SUBMIT, "c2"))])

        review = _reviewer(client).review(_ctx(shell))

        self.assertEqual("minor_issues_approve", review.classification)
        self.assertEqual("gemini:gemini-3.8-flash", review.model)
        self.assertEqual(["git log -1"], shell.commands)
        self.assertEqual(["submit_review", "shell"], [
            d.name for d in client.calls[0]["config"].tools[0].function_declarations])
        user, model_turn, results = client.calls[1]["contents"]
        self.assertIs(shell_turn.candidates[0].content, model_turn)
        response = results.parts[0].function_response
        self.assertEqual(("c1", "shell"), (response.id, response.name))
        self.assertIn("commit deadbeef", str(response.response))

    def test_empty_reply_is_a_provider_ended_turn(self) -> None:
        client = _ScriptedClient([types.GenerateContentResponse(candidates=[])])
        with self.assertRaises(ProviderTurnFailed):
            _reviewer(client).run(_ctx(None))

    def test_effort_and_summary_set_thinking(self) -> None:
        client = _ScriptedClient([_reply(_call("submit_review", SUBMIT, "c1"))])
        _reviewer(client, effort="high", reasoning_summary="detailed").review(_ctx(None))
        thinking = client.calls[0]["config"].thinking_config
        self.assertEqual("high", thinking.thinking_level.lower())
        self.assertTrue(thinking.include_thoughts)

    def test_unknown_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gemini_reviewer.GeminiReviewer("gemini-3.8-flash", name="g", effort="xhigh")


if __name__ == "__main__":
    unittest.main()
