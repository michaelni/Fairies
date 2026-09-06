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

llm_output_hacks: plain-text scope blocks end up in an HTML comment,
published branches and their tips become links."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_output_hacks import hide_scope_block, link_published_branch

GPT6_HEAD = """LLM-GPT-6-ASTRA — combined review of 1ce2a4db3060.

Scope GPT-5.6-SOL [code review]: Reported exhaustive review of every changed implementation, documentation, FATE-definition and reference hunk.

Scope GPT-5.6-SOL [design review]: Reported exhaustive changed-hunk review and deep review of metadata ordering and specifier semantics.

Scope GLM-5.3 [code review]: Reported deep review of the commit’s implementation, documentation and affected tests.

Scope combiner: Verified the specified commit, supplied discussion, related #21082 discussion, metadata helpers and call ordering.

### Moderate — [GPT-5.6-SOL] Format-level deletion does not distinguish which media type is re-encoded.
"""

HIDDEN_HEAD = """LLM-GPT-5.6-SOL combined review of 1ce2a4db3060.

<!--
Scope GPT-5.6-SOL code review: Exhaustive review of all changed hunks.
Scope combiner: Verified the exact PR head and merge base.
-->

### Moderate — first issue
"""


class HideScopeBlockTests(unittest.TestCase):
    def test_plain_scope_paragraphs_become_one_comment(self) -> None:
        result = hide_scope_block(GPT6_HEAD)
        self.assertEqual(result, """LLM-GPT-6-ASTRA — combined review of 1ce2a4db3060.

<!--
Scope GPT-5.6-SOL [code review]: Reported exhaustive review of every changed implementation, documentation, FATE-definition and reference hunk.
Scope GPT-5.6-SOL [design review]: Reported exhaustive changed-hunk review and deep review of metadata ordering and specifier semantics.
Scope GLM-5.3 [code review]: Reported deep review of the commit’s implementation, documentation and affected tests.
Scope combiner: Verified the specified commit, supplied discussion, related #21082 discussion, metadata helpers and call ordering.
-->

### Moderate — [GPT-5.6-SOL] Format-level deletion does not distinguish which media type is re-encoded.
""")

    def test_already_hidden_block_is_untouched(self) -> None:
        self.assertEqual(hide_scope_block(HIDDEN_HEAD), HIDDEN_HEAD)

    def test_message_without_scope_lines_is_untouched(self) -> None:
        text = "LLM-GPT-6-ASTRA\n\nNo issues found; the scope was the whole diff.\n"
        self.assertEqual(hide_scope_block(text), text)


SHA = "dc21a3ef5d6ecf45c7248bac4b2a6c8d1381dc20"
BRANCH_URL = "https://forge/Forgejo_Fairy/FFmpeg/src/branch/fairy/pr23842-test-ref"
COMMIT_URL = f"https://forge/Forgejo_Fairy/FFmpeg/commit/{SHA}"


def linked(message: str) -> str:
    return link_published_branch(message, "fairy/pr23842-test-ref", SHA,
                                 BRANCH_URL, COMMIT_URL)


class LinkPublishedBranchTests(unittest.TestCase):
    def test_bold_branch_and_abbreviated_tip_are_linked(self) -> None:
        self.assertEqual(
            linked("extended it on **fairy/pr23842-test-ref**, force-pushed as "
                   "dc21a3ef5d6e on top of 57ac82e08a73."),
            f"extended it on **[fairy/pr23842-test-ref]({BRANCH_URL})**, "
            f"force-pushed as [dc21a3ef5d6e]({COMMIT_URL}) on top of 57ac82e08a73.")

    def test_backticks_give_way_to_the_link(self) -> None:
        self.assertEqual(
            linked("see `fairy/pr23842-test-ref` (`dc21a3ef`)"),
            f"see [fairy/pr23842-test-ref]({BRANCH_URL}) ([dc21a3ef]({COMMIT_URL}))")

    def test_existing_links_and_urls_are_untouched(self) -> None:
        text = (f"[fairy/pr23842-test-ref]({BRANCH_URL}) and [dc21a3ef]({COMMIT_URL})"
                f" and {COMMIT_URL}")
        self.assertEqual(linked(text), text)

    def test_other_branches_and_hashes_are_untouched(self) -> None:
        text = "fairy/pr23842-test-ref-2, fairy/pr23842-test-reff, dc21a3, dc21a3ef5d6f"
        self.assertEqual(linked(text), text)

    def test_fenced_code_stays_verbatim(self) -> None:
        text = "```\ndc21a3ef5d6e fairy/pr23842-test-ref\n```\nsee dc21a3ef"
        self.assertEqual(
            linked(text),
            f"```\ndc21a3ef5d6e fairy/pr23842-test-ref\n```\nsee [dc21a3ef]({COMMIT_URL})")

    def test_hex_inside_the_branch_name_is_not_relinked(self) -> None:
        self.assertEqual(
            link_published_branch("on fairy/fix-dc21a3ef", "fairy/fix-dc21a3ef",
                                  SHA, "https://forge/b", COMMIT_URL),
            "on [fairy/fix-dc21a3ef](https://forge/b)")


if __name__ == "__main__":
    unittest.main()
