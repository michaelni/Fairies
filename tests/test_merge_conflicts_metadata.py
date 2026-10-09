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

The triage and the reviewer both see git's report of the conflicts
between the PR head and its target tip in the pull request metadata.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402

CONFLICTS = "Changelog\n\nAuto-merging Changelog\nCONFLICT (content): Merge conflict in Changelog"


class MergeConflictsMetadataTests(unittest.TestCase):
    def test_triage_and_review_show_the_conflicts(self) -> None:
        request = {"pull_request": {"number": 21037, "merge_conflicts": CONFLICTS}}
        for text in (llm_prompt.make_triage_user_text(request, False),
                     llm_prompt.make_user_text(request, [], [], False)):
            self.assertIn(f'"merge_conflicts": {json.dumps(CONFLICTS)}', text)

    def test_a_clean_merge_shows_none(self) -> None:
        self.assertNotIn("merge_conflicts", llm_prompt.make_triage_user_text(
            {"pull_request": {"number": 22961}}, False))


if __name__ == "__main__":
    unittest.main()
