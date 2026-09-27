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

from unittest import mock  # noqa: E402

import httpx  # noqa: E402

import common  # noqa: E402
import gemini_reviewer  # noqa: E402
from llm_review_api import ProviderTurnFailed  # noqa: E402
from tests.test_anthropic_reviewer import _ctx, _FakeShell  # noqa: E402

SUBMIT = {"classification": "minor_issues_approve", "message": "LLM review: one nit.",
          "head_vs_branch_diff_evidence": False}


def _reply(*calls: tuple[str, dict, str]) -> dict:
    return {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
        {"functionCall": {"name": name, "args": args, "id": call_id},
         "thoughtSignature": "c2ln"} for name, args, call_id in calls]}}]}


class _ScriptedClient:
    """Answers each POST with the next queued (status, JSON body)."""

    def __init__(self, replies: list[tuple[int, dict]]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def post(self, url: str, *, json: dict) -> httpx.Response:
        self.bodies.append(json)
        status, body = self.replies.pop(0)
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))


def _reviewer(client: _ScriptedClient, **kwargs: object) -> gemini_reviewer.GeminiReviewer:
    reviewer = gemini_reviewer.GeminiReviewer(
        "gemini-3.8-flash", name="gemini:gemini-3.8-flash", **kwargs)
    reviewer._client = lambda: client  # type: ignore[method-assign]
    return reviewer


class GeminiReviewLoopTests(unittest.TestCase):
    def test_shell_round_then_submit(self) -> None:
        shell = _FakeShell()
        shell_turn = _reply(("shell", {"command": "git log -1"}, "c1"))
        client = _ScriptedClient([(200, shell_turn),
                                  (200, _reply(("submit_review", SUBMIT, "c2")))])

        review = _reviewer(client).review(_ctx(shell))

        self.assertEqual("minor_issues_approve", review.classification)
        self.assertEqual("gemini:gemini-3.8-flash", review.model)
        self.assertEqual(["git log -1"], shell.commands)
        self.assertEqual(["submit_review", "shell"], [
            d["name"] for d in client.bodies[0]["tools"][0]["functionDeclarations"]])
        user, model_turn, results = client.bodies[1]["contents"]
        self.assertEqual(shell_turn["candidates"][0]["content"], model_turn)
        response = results["parts"][0]["functionResponse"]
        self.assertEqual(("c1", "shell"), (response["id"], response["name"]))
        self.assertIn("commit deadbeef", str(response["response"]))

    def test_rate_limit_is_retried(self) -> None:
        client = _ScriptedClient([(429, {"error": {"code": 429}}),
                                  (200, _reply(("submit_review", SUBMIT, "c1")))])
        with mock.patch.object(common.time, "sleep"):
            review = _reviewer(client).review(_ctx(None))
        self.assertEqual("minor_issues_approve", review.classification)
        self.assertEqual(2, len(client.bodies))

    def test_empty_reply_is_a_provider_ended_turn(self) -> None:
        client = _ScriptedClient([(200, {"promptFeedback": {"blockReason": "OTHER"}})])
        with self.assertRaises(ProviderTurnFailed):
            _reviewer(client).run(_ctx(None))

    def test_effort_and_summary_set_thinking(self) -> None:
        client = _ScriptedClient([(200, _reply(("submit_review", SUBMIT, "c1")))])
        _reviewer(client, effort="high", reasoning_summary="detailed").review(_ctx(None))
        self.assertEqual({"includeThoughts": True, "thinkingLevel": "high"},
                         client.bodies[0]["generationConfig"]["thinkingConfig"])

    def test_unknown_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gemini_reviewer.GeminiReviewer("gemini-3.8-flash", name="g", effort="xhigh")


if __name__ == "__main__":
    unittest.main()
