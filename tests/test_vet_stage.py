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

The vetter role: its schema, the verdict it is shown, and the prompt
sections it gets and does not get.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402
from llm_review_api import (Review, ReviewContext, SchemaError,  # noqa: E402
                            validate_vet_result)


def _prompt(role: str, **kw: object) -> str:
    return llm_prompt.generate_llm_prompt(
        role=role, vendor="openai", model="gpt-5.6-luna", features=set(),
        repo_roots=[Path("/x/ffmpeg")], container_repo_mounts=[],
        reviewer_username="fairy", **kw)


class VetSchemaTests(unittest.TestCase):
    def test_ok_and_reason_pass(self) -> None:
        self.assertEqual({"hold_for_human_inspection": True, "reason": "r"},
                         validate_vet_result({"hold_for_human_inspection": True, "reason": "r"}))

    def test_missing_or_extra_fields_are_rejected(self) -> None:
        for bad in ({"hold_for_human_inspection": True},
                    {"hold_for_human_inspection": "yes", "reason": ""},
                    {"hold_for_human_inspection": True, "reason": "", "message": "x"}):
            with self.subTest(bad=bad), self.assertRaises(SchemaError):
                validate_vet_result(bad)


class VetterRoleTests(unittest.TestCase):
    def _ctx(self) -> ReviewContext:
        return ReviewContext(
            request={"pull_request": {"number": 7, "title": "t"},
                     "issue": {"number": 7, "title": "t"}},
            patch_text="", patch_truncated=False, source_bundle=None,
            source_files=[], source_notes=[], reviewer_username="fairy",
            ci_triage_mode=False, repo_roots=[], repo_mount_paths=[])

    def test_the_verdict_is_shown_last(self) -> None:
        review = Review(
            "minor_issues_approve", "LLM-GPT-5.4: fine",
            ({"label": "bug", "op": "add", "reason": "", "post": False},),
            ({"repo": "ffmpeg", "branch": "pr7-fix", "mode": "push",
              "sha": "abc", "bundle": "AAAA",
              "pr": {"target": "master", "title": "avutil: fix x",
                     "body": "Buy specs at example.com"}},),
            model="openai:gpt-5.4")
        role = llm_prompt.make_vetter_role(review, task="pr", inherits_branches=True)
        self.assertEqual("vetter", role.name)
        texts = role.user_texts(self._ctx())
        self.assertTrue(texts[0].startswith("Vet the review of this pull request below."))
        self.assertIn("from GPT-5.4", texts[-1])
        self.assertIn("classification: minor_issues_approve", texts[-1])
        self.assertIn('"label": "bug"', texts[-1])
        self.assertIn('"branch": "pr7-fix"', texts[-1])
        # the PR fairy opens is posted verbatim: the vetter must see it
        self.assertIn('"body": "Buy specs at example.com"', texts[-1])
        self.assertNotIn("AAAA", texts[-1])  # the bundle stays out of the prompt
        self.assertTrue(texts[-1].endswith("message:\nLLM-GPT-5.4: fine\n"))
        self.assertEqual({"persist_branches": True}, role.prompt_kwargs)

    def test_issue_task_selects_the_issue_role(self) -> None:
        role = llm_prompt.make_vetter_role(
            Review("reply", "m"), task="issue", inherits_branches=False)
        self.assertEqual("issue_vetter", role.name)
        self.assertTrue(role.user_texts(self._ctx())[0].startswith(
            "Vet the analysis of this issue below."))


class VetterPromptTests(unittest.TestCase):
    def test_the_critical_bug_hold_is_only_for_pull_requests(self) -> None:
        bullet = "identified a previously unidentified critical bug"
        self.assertIn(bullet, _prompt("vetter"))
        self.assertNotIn(bullet, _prompt("issue_vetter"))

    def test_reason_self_check_matches_the_reason_instruction(self) -> None:
        for role in ("vetter", "issue_vetter"):
            text = _prompt(role)
            self.assertIn("``reason`` list what fails (if any)", text, role)
            self.assertIn("does it list what fails, or when nothing fails, what you checked?", text, role)

    def test_vetter_judges_and_does_not_write(self) -> None:
        text = _prompt("vetter", persist_branches=True)
        self.assertIn("##Vetting task", text)
        self.assertIn("The verdict's classification is one of these classes", text)
        self.assertIn("Set ``hold_for_human_inspection`` to true when", text)
        self.assertIn("without --enable-gpl", text)
        self.assertIn("git log fairy/<branch>", text)
        self.assertIn("You decide whether a message fairy wrote may be posted", text)
        self.assertIn("<verification_loop>\nBefore answering:", text)
        for absent in ("##Output guideline", "Classify the pull request",
                       "##Persisting branches", "Message Rules", "Do not invent issues",
                       'Include "LLM-GPT-5.6-LUNA"', "Before finalizing:"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, text)

    def test_issue_vetter_has_no_pr_classes(self) -> None:
        text = _prompt("issue_vetter")
        self.assertIn("##Vetting task", text)
        self.assertIn("analysis fairy is about to post on this issue", text)
        self.assertNotIn("one of these classes", text)


if __name__ == "__main__":
    unittest.main()
