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

The self-identification each role's prompt asks for names one model once.

Posted roles (combiner, triagers, issue investigator) open with
``LLM-<label>`` (vendor prefix stripped, uppercased). The PR reviewer is a
combiner-bound draft: it identifies with the bare label the combiner's
"Draft review from <label>" headers already use — asking drafts for the
``LLM-`` form made combined reviews mix "LLM-GLM-5.2" with "GLM-5.2" for
the same model (e.g. PR #23016).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402
from llm_review_api import Review  # noqa: E402


class ModelLabelTests(unittest.TestCase):
    def test_vendor_prefix_stripped_and_uppercased(self) -> None:
        self.assertEqual("GLM-5.2", llm_prompt.model_label("zai:glm-5.2"))
        self.assertEqual("GPT-5.4", llm_prompt.model_label("gpt-5.4"))
        self.assertEqual("UNKNOWN", llm_prompt.model_label(""))


class PromptForTests(unittest.TestCase):
    def test_derived_identity_facts(self) -> None:
        for role, subject, persona, combiner, draft in (
            ("review",             "PR",    "reviewer",     False, True),
            ("code_review",        "PR",    "reviewer",     False, True),
            ("design_review",      "PR",    "reviewer",     False, True),
            ("combiner",           "PR",    "reviewer",     True,  False),
            ("triager",            "PR",    "reviewer",     False, False),
            ("issue_investigator", "issue", "investigator", False, False),
            ("issue_combiner",     "issue", "investigator", True,  False),
            ("issue_triager",      "issue", "investigator", False, False),
        ):
            ctx = llm_prompt.PromptFor(role, "gpt-5.4")
            self.assertEqual(
                (subject, persona, combiner, draft),
                (ctx.subject, ctx.persona, ctx.combiner, ctx.draft),
                role,
            )


class LlmPrefixPromptTests(unittest.TestCase):
    def _prompt(self, role: str, model: str) -> str:
        return llm_prompt.generate_llm_prompt(
            role=role, vendor="openai", model=model, features=set(),
            repo_roots=[], container_repo_mounts=[], reviewer_username="fairy",
        )

    def test_posted_roles_identify_with_the_llm_prefix(self) -> None:
        for role, model, label in (
            ("combiner", "gpt-5.4", "LLM-GPT-5.4"),
            ("triager", "gpt-5.4-mini", "LLM-GPT-5.4-MINI"),
            ("issue_investigator", "zai:glm-5.2", "LLM-GLM-5.2"),
            ("issue_combiner", "gpt-5.4", "LLM-GPT-5.4"),
            ("issue_triager", "gpt-5.4-mini", "LLM-GPT-5.4-MINI"),
        ):
            prompt = self._prompt(role, model)
            self.assertIn(f'Include "{label}"', prompt, role)

    def test_draft_reviewer_identifies_with_the_bare_label(self) -> None:
        for role in llm_prompt.REVIEW_PROMPTS:
            prompt = self._prompt(role, "zai:glm-5.2")
            self.assertIn('Include "GLM-5.2"', prompt, role)
            self.assertNotIn('Include "LLM-', prompt, role)

    def test_combiner_user_text_uses_the_same_labels(self) -> None:
        text = llm_prompt.make_combiner_user_text([
            Review("minor_issues_approve", "a", model="openai:gpt-5.4"),
            Review("major_issues", "b", model="zai:glm-5.2"),
        ])
        self.assertIn("Draft review from GPT-5.4", text)
        self.assertIn("Draft review from GLM-5.2", text)

    def test_combiner_user_text_separates_one_model_two_prompts(self) -> None:
        """Without the prompt in the header the combiner cannot attribute
        an issue to the draft that raised it, nor tell the two apart."""
        text = llm_prompt.make_combiner_user_text([
            Review("minor_issues_approve", "a", model="codex:gpt-5.6-sol",
                   prompt="code_review"),
            Review("major_issues", "b", model="codex:gpt-5.6-sol",
                   prompt="design_review"),
        ])
        self.assertIn("Draft code review from GPT-5.6-SOL", text)
        self.assertIn("Draft design review from GPT-5.6-SOL", text)

    def test_each_split_prompt_carries_only_its_own_role(self) -> None:
        code = self._prompt("code_review", "gpt-5.4")
        design = self._prompt("design_review", "gpt-5.4")
        both = self._prompt("review", "gpt-5.4")
        for text in (code, both):
            self.assertIn("##In your Code Reviewer role", text)
            self.assertIn("Suggest to add tests", text)
            self.assertIn("* NULL pointer dereference.", text)
            self.assertIn("* Signed integer overflows", text)
        for text in (design, both):
            self.assertIn("##In your Design Reviewer role", text)
            self.assertIn("suggest factorizations", text)
        self.assertNotIn("##In your Design Reviewer role", code)
        self.assertNotIn("##In your Code Reviewer role", design)
        self.assertNotIn("Suggest to add tests", design)
        self.assertNotIn("NULL pointer dereference", design)
        self.assertNotIn("Signed integer overflows", design)

    def test_the_split_prompts_keep_the_shared_bullets(self) -> None:
        """The design reviewer has no code reviewer section to carry them,
        so they must be spliced into its own."""
        for role in llm_prompt.REVIEW_PROMPTS:
            prompt = self._prompt(role, "gpt-5.4")
            self.assertIn("Do not present stylistic preferences", prompt, role)
            self.assertIn("state the scope and depth of the review", prompt, role)
            self.assertIn("##In your project assistant role.", prompt, role)
            self.assertIn("Inspect related parts of specifications", prompt, role)
            self.assertIn("Additional Major issues:", prompt, role)
            self.assertIn("- approve: no substantive issues", prompt, role)
            self.assertEqual(1, prompt.count("- review / check each commit."), role)

    def test_the_combiner_may_not_drop_a_design_point_as_style(self) -> None:
        """It has no design mandate of its own and is told twice to drop
        stylistic preferences, so a draft made of design findings would
        otherwise be the easiest thing in its input to discard."""
        self.assertIn("not a stylistic preference",
                      self._prompt("combiner", "gpt-5.4"))

    def test_an_issue_draft_keeps_the_bare_review_header(self) -> None:
        text = llm_prompt.make_combiner_user_text([
            Review("reply", "a", model="zai:glm-5.2", prompt="issue_investigator"),
        ])
        self.assertIn("Draft review from GLM-5.2", text)


if __name__ == "__main__":
    unittest.main()
