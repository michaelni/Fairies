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

llm_output_hacks: plain-text scope blocks end up in an HTML comment."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_output_hacks import hide_scope_block

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


if __name__ == "__main__":
    unittest.main()
