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
published branches and their tips become links, bold list entries and
the dropped-claims and verification sections fold into details blocks."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_output_hacks import (
    fold_entries, fold_sections, hide_scope_block, link_published_branch,
    names_published_branch,
)

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
PUSHED_NAME_MESSAGE = ("I pushed `issue24248-bugfix` to the FFmpeg fairy remote for the"
                       " proposed PR. I also pushed `issue24248-tooling` to the fairy remote.")


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
                f" and {COMMIT_URL} and [the fairy/pr23842-test-ref branch]({BRANCH_URL})")
        self.assertEqual(linked(text), text)

    def test_bracket_ahead_of_a_link_is_not_part_of_it(self) -> None:
        self.assertEqual(
            linked("video at [48, 53418363) on fairy/pr23842-test-ref."
                   " MOV's [packet-selection logic](https://forge/mov)"),
            f"video at [48, 53418363) on [fairy/pr23842-test-ref]({BRANCH_URL})."
            " MOV's [packet-selection logic](https://forge/mov)")

    def test_code_span_holding_a_mention_is_linked_as_a_whole(self) -> None:
        for span, url in (("`git log fairy/pr23842-test-ref -3`", BRANCH_URL),
                          ("`git show 57ac82e08a73 dc21a3ef`", COMMIT_URL),
                          ("`git log fairy/pr23842-test-ref dc21a3ef`", BRANCH_URL),
                          ("``git log `fairy/pr23842-test-ref` ``", BRANCH_URL)):
            with self.subTest(span=span):
                self.assertEqual(linked(f"run {span} now"), f"run [{span}]({url}) now")

    def test_other_code_spans_and_linked_ones_are_untouched(self) -> None:
        text = ("`git log master`, [`git log fairy/pr23842-test-ref`](https://forge/log),"
                " [`a[0]` on ` fairy/pr23842-test-ref here](https://forge/log)")
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

    def test_backticked_pushed_name_links_as_the_forge_name(self) -> None:
        self.assertEqual(
            link_published_branch(PUSHED_NAME_MESSAGE, "fairy/issue24248-bugfix",
                                  SHA, "https://forge/b", COMMIT_URL),
            "I pushed [fairy/issue24248-bugfix](https://forge/b) to the FFmpeg"
            " fairy remote for the proposed PR. I also pushed `issue24248-tooling`"
            " to the fairy remote.")


class NamesPublishedBranchTests(unittest.TestCase):
    def test_backticked_pushed_name_counts(self) -> None:
        self.assertTrue(
            names_published_branch(PUSHED_NAME_MESSAGE, "fairy/issue24248-tooling"))

    def test_bare_backticked_bold_and_linked_names_count(self) -> None:
        for text in ("on fairy/pr23842-test-ref.", "see `fairy/pr23842-test-ref`",
                     "**fairy/pr23842-test-ref**",
                     f"[fairy/pr23842-test-ref]({BRANCH_URL})"):
            with self.subTest(text=text):
                self.assertTrue(names_published_branch(text, "fairy/pr23842-test-ref"))

    def test_other_branches_urls_and_fenced_code_do_not(self) -> None:
        for text in ("fairy/pr23842-test-ref-2", "fairy/pr23842-test-reff",
                     "pr23842-test-ref", BRANCH_URL,
                     "```\nfairy/pr23842-test-ref\n```"):
            with self.subTest(text=text):
                self.assertFalse(names_published_branch(text, "fairy/pr23842-test-ref"))


if __name__ == "__main__":
    unittest.main()


FINDING = """### Findings
1. **Moderate — [GPT-6-ASTRA code/design] The D3D11 path ignores `AV_HWFRAME_MAP_DIRECT`.** bdb29eff08a9, [libavutil/hwcontext_amf.c:951](https://code.ffmpeg.org/FFmpeg/FFmpeg/src/commit/bdb29eff08a91db943e17be20655c2fda7385eb3/libavutil/hwcontext_amf.c#L951).
   The helper receives no mapping flags and unconditionally calls `Convert(AMF_MEMORY_DX11)`.

   ```c
   if (flags & AV_HWFRAME_MAP_DIRECT)
       return AVERROR(ENOSYS);
   ```
   The probe confirms that this rejects host-memory direct mapping.

### Verification
- The [hardware-configuration contract](https://example.org/c) requires a frames context.
"""

FOLDED_FINDING = """### Findings
1. <details><summary><b>Moderate — [GPT-6-ASTRA code/design] The D3D11 path ignores <code>AV_HWFRAME_MAP_DIRECT</code>.</b> bdb29eff08a9, <a href="https://code.ffmpeg.org/FFmpeg/FFmpeg/src/commit/bdb29eff08a91db943e17be20655c2fda7385eb3/libavutil/hwcontext_amf.c#L951">libavutil/hwcontext_amf.c:951</a>.</summary>

   The helper receives no mapping flags and unconditionally calls `Convert(AMF_MEMORY_DX11)`.

   ```c
   if (flags & AV_HWFRAME_MAP_DIRECT)
       return AVERROR(ENOSYS);
   ```
   The probe confirms that this rejects host-memory direct mapping.
   </details>

### Verification
- The [hardware-configuration contract](https://example.org/c) requires a frames context.
"""

ONE_LINERS = """- **GROK-4.7 (code/design): Untested P010 is not evidence of broken P010 support.** Its format mapping and acceptance are consistent with the code.
- **Classification: major_issues.**
"""

FOLDED_ONE_LINERS = """- <details><summary><b>GROK-4.7 (code/design): Untested P010 is not evidence of broken P010 support.</b></summary>

  Its format mapping and acceptance are consistent with the code.
  </details>
- **Classification: major_issues.**
"""


class TestFoldEntries(unittest.TestCase):
    def test_bold_entry_with_a_body_folds_and_its_first_line_becomes_html(self) -> None:
        self.assertEqual(fold_entries(FINDING), FOLDED_FINDING)

    def test_one_line_entry_splits_after_its_bold_lead(self) -> None:
        self.assertEqual(fold_entries(ONE_LINERS), FOLDED_ONE_LINERS)

    def test_fenced_code_at_the_margin_is_untouched(self) -> None:
        fenced = "```\n1. **not** an entry\n   body\n```\n"
        self.assertEqual(fold_entries(fenced), fenced)

    def test_summary_escapes_html_and_keeps_italics(self) -> None:
        self.assertEqual(
            fold_entries("1. **Minor:** a < b, *every* `x<y>`\n   body\n"),
            "1. <details><summary><b>Minor:</b> a &lt; b, <i>every</i> <code>x&lt;y&gt;</code></summary>"
            "\n\n   body\n   </details>\n")


SECTIONS = """### Findings
1. **Minor — the commit message overstates the effect of `reconnect_at_eof`.** b5f143807b35

### Dropped claims
1. **GROK-4.7 (code): drop the claim that a whole-body `200` response to a Range request is illegal.** RFC 9110 permits it.

**Verification**
My seekable disconnect test cut the second segment at byte 8,000: the parent read 106 packets, the head 24.

```sh
# the playlist used
./ffmpeg -i http://127.0.0.1:8000/vod.m3u8 -f null -
```

I verified and retained **`fairy/pr25148-tooling`** at 9d1b4831c17d.

**Classification: major_issues.** The verified finite-input replay makes the current code worse if merged.
"""

FOLDED_SECTIONS = """### Findings
1. **Minor — the commit message overstates the effect of `reconnect_at_eof`.** b5f143807b35

<details><summary><b>Dropped claims</b></summary>

1. **GROK-4.7 (code): drop the claim that a whole-body `200` response to a Range request is illegal.** RFC 9110 permits it.
</details>

<details><summary><b>Verification</b></summary>

My seekable disconnect test cut the second segment at byte 8,000: the parent read 106 packets, the head 24.

```sh
# the playlist used
./ffmpeg -i http://127.0.0.1:8000/vod.m3u8 -f null -
```

I verified and retained **`fairy/pr25148-tooling`** at 9d1b4831c17d.
</details>

**Classification: major_issues.** The verified finite-input replay makes the current code worse if merged.
"""


class TestFoldSections(unittest.TestCase):
    def test_named_sections_fold_up_to_the_next_heading_or_bold_paragraph(self) -> None:
        self.assertEqual(fold_sections(SECTIONS), FOLDED_SECTIONS)
        self.assertEqual(fold_sections("### Verification\nran fate\n### Replies\n"),
                         "<details><summary><b>Verification</b></summary>\n\nran fate\n</details>\n### Replies\n")

    def test_other_sections_and_a_message_without_headings_are_untouched(self) -> None:
        self.assertEqual(fold_sections(HIDDEN_HEAD), HIDDEN_HEAD)
        self.assertEqual(fold_sections("plain text\n"), "plain text\n")
