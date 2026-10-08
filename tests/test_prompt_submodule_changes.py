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

The review prompts list ``pull_request.submodule_changes``, so the model
does not have to spot a one-line ``Subproject commit ...`` hunk hidden
inside a multi-thousand-line diff.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402


ADDED = {"path": "tests/checkasm/ext", "action": "added", "old_commit": None,
         "new_commit": "deadbeefcafebabe1234567890abcdef00112233"}
UPDATED = {"path": "tests/checkasm/ext", "action": "updated",
           "old_commit": "aaaaaaa1111111122222222333333334444444455",
           "new_commit": "bbbbbbb1111111122222222333333334444444455"}


def _make_request(*, submodule_changes: list[dict], number: int) -> dict:
    return {
        "pull_request": {
            "number": number,
            "title": "test",
            "author": "alice",
            "html_url": f"https://example.org/pr/{number}",
            "base_ref": "master",
            "head_ref": "alice/test",
            "head_sha": "f" * 40,
            "submodule_changes": submodule_changes,
            "additions": 1,
            "deletions": 0,
            "changed_files": 1,
            "auto_merge": None,
            "body": "",
        },
        "patch": "",
        "patch_truncated": False,
        "discussion": [],
        "reviewer_username": "Forgejo_Fairy",
    }


class MakeUserTextSubmoduleMetadataTests(unittest.TestCase):
    def test_make_user_text_surfaces_submodule_changes(self) -> None:
        text = llm_prompt.make_user_text(
            _make_request(submodule_changes=[ADDED], number=12345),
            source_notes=[],
            source_files=[".gitmodules", "tests/regular.c"],
            patch_was_truncated=False,
        )
        self.assertIn('"submodule_changes":', text)
        self.assertIn('"path": "tests/checkasm/ext"', text)
        self.assertIn('"action": "added"', text)
        self.assertIn('"new_commit": "deadbeefcafebabe1234567890abcdef00112233"', text)

    def test_make_user_text_omits_submodule_changes_for_normal_pr(self) -> None:
        text = llm_prompt.make_user_text(
            _make_request(submodule_changes=[], number=12346),
            source_notes=[],
            source_files=["x.c"],
            patch_was_truncated=False,
        )
        self.assertNotIn("submodule_changes", text)


class MakeTriageUserTextSubmoduleMetadataTests(unittest.TestCase):
    """The triage gatekeeper sees the same submodule callout, since
    deciding whether to engage the main reviewer is exactly when the
    "external code is being pulled in" signal matters most."""

    def test_make_triage_user_text_surfaces_submodule_changes(self) -> None:
        text = llm_prompt.make_triage_user_text(
            _make_request(submodule_changes=[UPDATED], number=22000),
            patch_was_truncated=False,
        )
        self.assertIn('"submodule_changes":', text)
        self.assertIn('"action": "updated"', text)
        self.assertIn('"old_commit": "aaaaaaa1111111122222222333333334444444455"', text)
        self.assertIn('"new_commit": "bbbbbbb1111111122222222333333334444444455"', text)


if __name__ == "__main__":
    unittest.main()
