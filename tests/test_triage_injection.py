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

Triage prompt-injection flag: schema field, prompt wiring, forced skip.

The triage model reads every attacker-controllable byte of a PR (title,
description, comments, patch), so it doubles as the injection detector:
``prompt_injection: true`` forces route=skip in ``validate_triage_result``
regardless of the route the model chose -- the injected text may have
steered the route itself.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402
import llm_review_api  # noqa: E402
import pr_review_wrapper as wrapper  # noqa: E402


class TriageInjectionTests(unittest.TestCase):
    def test_flag_forces_skip_for_every_route(self) -> None:
        for route in ("engage", "reply_no_verdict", "skip"):
            with self.subTest(route=route):
                result = llm_review_api.validate_triage_result({
                    "route": route,
                    "message": "hi" if route == "reply_no_verdict" else "",
                    "reason": "PR description tells the AI to approve",
                    "prompt_injection": True,
                })
                self.assertEqual("skip", result["route"])
                self.assertEqual("", result["message"])
                self.assertEqual(
                    f"suspected prompt injection (triage chose {route}): "
                    "PR description tells the AI to approve",
                    result["reason"])

    def test_forced_skip_reason_leads_with_the_suspicion(self) -> None:
        """PR 23680 (2026-08-01): triage chose engage with prompt_injection
        true and a reason whose first 150 chars only explained why a review
        was warranted; the ticket showed that truncated engage rationale as
        the skip reason. The rewritten reason must surface the injection
        before any truncation can eat the tail."""
        result = llm_review_api.validate_triage_result({
            "route": "engage",
            "message": "",
            "reason": (
                "Forgejo_Fairy has never reviewed this pull request, and the "
                "author force-pushed new code before the latest activity, so "
                "a full reviewer pass is warranted. The latest author comment "
                "also contains an explicit instruction to the reviewing AI "
                "not to review and attempts to manipulate the review outcome."
            ),
            "prompt_injection": True,
        })
        self.assertEqual("skip", result["route"])
        self.assertTrue(result["reason"].startswith(
            "suspected prompt injection (triage chose engage): "))

    def test_unflagged_result_is_untouched(self) -> None:
        result = llm_review_api.validate_triage_result({
            "route": "engage", "message": "", "reason": "ok",
            "prompt_injection": False,
        })
        self.assertEqual("engage", result["route"])

    def test_schema_requires_the_flag(self) -> None:
        schema = llm_review_api.build_triage_schema([])["schema"]
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {"route": "skip", "message": "", "reason": "ok"}, schema,
            )
        llm_review_api.check_schema(
            {"route": "skip", "message": "", "reason": "ok",
             "prompt_injection": False, "requested_verbosity": None}, schema,
        )

    def test_prompt_documents_the_field(self) -> None:
        prompt = llm_prompt.make_triage_developer_prompt(
            "fairy", [Path("ffmpeg")], False, False, False, False, [],
            ctx=llm_prompt.PromptFor("triager", "gpt-5.4-mini"),
        )
        self.assertIn("prompt_injection", prompt)

    def test_triager_role_binds_schema_and_validator(self) -> None:
        role = llm_prompt.make_triager_role(allowed_models=["gpt-5.4"], allowed_labels=[])
        result = role.validate({
            "route": "engage", "message": "", "reason": "ok",
            "prompt_injection": False, "requested_verbosity": None,
            "requested_models": [], "requested_effort": None,
        })
        self.assertEqual("engage", result["route"])


if __name__ == "__main__":
    unittest.main()
