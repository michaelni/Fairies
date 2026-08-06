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

Pins the CI-failure-data wiring in ``llm_prompt``.

The reviewer/combiner developer prompt carries the ``log_tail`` /
failure-listing guidance only when the head CI is red; the triager has
no CI-mode prompt at all (the CI-red announcement path was removed) and
sees the ``ci_triage`` payload in its user message instead. These tests
guard that wiring so the reviewer does not silently lose (or always
carry) the CI failure context and no engage-blocking CI mode creeps
back into the triager.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402


def _prompt(role: str, *, ci_triage_mode: bool) -> str:
    return llm_prompt.generate_llm_prompt(
        role=role,
        vendor="openai",
        model="m",
        features=set(),
        repo_roots=[],
        container_repo_mounts=[],
        reviewer_username="fairy",
        ci_triage_mode=ci_triage_mode,
    )


def _reviewer_prompt(*, ci_failures_present: bool) -> str:
    return _prompt("review", ci_triage_mode=ci_failures_present)


class CiFailureDataFactoringTests(unittest.TestCase):
    def test_reviewer_gets_ci_data_only_on_red_ci(self) -> None:
        self.assertIn("log_tail", _reviewer_prompt(ci_failures_present=True))
        self.assertNotIn("log_tail", _reviewer_prompt(ci_failures_present=False))

    def test_triager_prompt_has_no_ci_mode(self) -> None:
        # Red CI must not steer the triager away from ``engage``: the
        # developer prompt is identical for green and red heads.
        self.assertEqual(
            _prompt("triager", ci_triage_mode=True),
            _prompt("triager", ci_triage_mode=False),
        )
        self.assertNotIn(
            "CI failure mode", _prompt("triager", ci_triage_mode=True))

    def test_ci_data_is_dumped_into_the_triage_user_text(self) -> None:
        request = {
            "pull_request": {"number": 1},
            "ci_triage": {
                "head_sha": "h1",
                "failure_contexts": [
                    {"context": "/ build", "log_tail": "make: *** error"}],
            },
        }
        text = llm_prompt.make_triage_user_text(request, False)
        self.assertIn("make: *** error", text)
        self.assertIn("CI failure data from the caller", text)


if __name__ == "__main__":
    unittest.main()
