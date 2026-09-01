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

Ungraded drafts under a configured combiner.

The combiner alone grades the pull request, so a draft feeding it emits
no classification and its prompt keeps the concrete issue cases without
the severity buckets and without any classification instructions --
severity wording in a draft would predispose the combiner's grading.
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


class UngradedSchemaTests(unittest.TestCase):
    def test_validator_fills_in_the_ungraded_marker(self) -> None:
        result = llm_review_api.validate_ungraded_review(
            {"message": "m", "head_vs_branch_diff_evidence": False})
        self.assertEqual(llm_review_api.UNGRADED, result["classification"])

    def test_an_emitted_classification_is_rejected(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.validate_ungraded_review(
                {"classification": "approve", "message": "",
                 "head_vs_branch_diff_evidence": False})

    def test_role_binds_the_ungraded_schema(self) -> None:
        role = llm_prompt.UNGRADED_REVIEWER_ROLE
        self.assertNotIn(
            "classification", role.schema["schema"]["properties"])
        self.assertIs(llm_review_api.validate_ungraded_review, role.validate)
        self.assertEqual({"classifies": False}, role.prompt_kwargs)

    def test_labels_wrap_the_ungraded_validator(self) -> None:
        role = llm_prompt.role_with_labels(
            llm_prompt.UNGRADED_REVIEWER_ROLE, ["bug"])
        result = role.validate(
            {"message": "m", "head_vs_branch_diff_evidence": False,
             "label_changes": []})
        self.assertEqual(llm_review_api.UNGRADED, result["classification"])


def prompt_for(role: str, classifies: bool = True) -> str:
    return llm_prompt.generate_llm_prompt(
        role=role, vendor="openai", model="gpt-5.6", features=set(),
        repo_roots=[Path("ffmpeg")], container_repo_mounts=[],
        reviewer_username="fairy", classifies=classifies)


class UngradedPromptTests(unittest.TestCase):
    def test_draft_prompt_carries_no_severity_vocabulary(self) -> None:
        text = prompt_for("review", classifies=False)
        for grading in ("Classify the pull request",
                        "minor_issues_approve", "moderate_issues",
                        "major_issues", "Additional Minor issues",
                        "Additional Moderate issues",
                        "Additional Major issues"):
            self.assertNotIn(grading, text)
        self.assertIn("do not classify or rank the issues by severity", text)

    def test_draft_prompt_keeps_the_issue_cases(self) -> None:
        text = prompt_for("review", classifies=False)
        for case in ("Unrelated changes should be in separate patches.",
                     "Public API should be documented.",
                     "* Out of array access.",
                     "Non issues:"):
            self.assertIn(case, text)

    def test_reviewer_and_combiner_keep_the_classes_when_grading(self) -> None:
        for role in ("review", "combiner"):
            text = prompt_for(role)
            self.assertIn("Classify the pull request", text, role)
            self.assertIn("Additional Major issues:", text, role)


if __name__ == "__main__":
    unittest.main()
