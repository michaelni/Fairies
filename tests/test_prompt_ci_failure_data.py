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

Pins the CI-failure-data factoring in ``llm_prompt``.

The ``log_tail`` / failure-listing guidance is shared by the triager and
the reviewer: the triage CI-mode prompt is built on top of it, and the
reviewer prompt includes it only when the head CI is red. These tests
guard that wiring so the reviewer does not silently lose (or always
carry) the CI failure context.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402


def _reviewer_prompt(*, ci_failures_present: bool) -> str:
    return llm_prompt.generate_llm_prompt(
        role="review",
        vendor="openai",
        model="m",
        features=set(),
        repo_roots=[],
        container_repo_mounts=[],
        reviewer_username="fairy",
        ci_triage_mode=ci_failures_present,
    )


class CiFailureDataFactoringTests(unittest.TestCase):
    def test_triage_ci_mode_is_built_on_shared_snippet(self) -> None:
        self.assertTrue(
            llm_prompt.T_PROMPT_TRIAGE_CI_MODE.startswith(
                llm_prompt.CRT_PROMPT_CI_FAILURE_DATA
            )
        )
        # The triage-only routing guidance still rides along.
        self.assertIn("SHOULD NOT choose engage", llm_prompt.T_PROMPT_TRIAGE_CI_MODE)

    def test_reviewer_gets_ci_data_only_on_red_ci(self) -> None:
        self.assertIn("log_tail", _reviewer_prompt(ci_failures_present=True))
        self.assertNotIn("log_tail", _reviewer_prompt(ci_failures_present=False))

    def test_reviewer_ci_data_omits_triage_routing(self) -> None:
        # The reviewer engages a real review; the triage route guidance
        # ("SHOULD NOT choose engage") must not leak into its prompt.
        self.assertNotIn(
            "SHOULD NOT choose engage", _reviewer_prompt(ci_failures_present=True)
        )


if __name__ == "__main__":
    unittest.main()
