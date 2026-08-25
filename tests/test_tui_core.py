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

tui_core: grid layout math, block tiler, ring buffer, markdown renderer."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tui_core  # noqa: E402
from tui_core import GridLayout, Rect, RingBuffer, render_markdown  # noqa: E402


def line_text(line: tui_core.StyledLine) -> str:
    return "".join(t for _, t in line)


def styles(lines: list[tui_core.StyledLine]) -> set[str]:
    return {s for line in lines for s, _ in line}


class GridLayoutTests(unittest.TestCase):
    def test_rects_tile_the_area_around_the_dividers(self) -> None:
        g = GridLayout(fx_top=0.5, fx_bottom=0.5, fy=0.4)
        r = g.rects(100, 40)
        self.assertEqual(r["tl"], Rect(0, 0, 50, 16))
        self.assertEqual(r["tr"], Rect(51, 0, 49, 16))
        self.assertEqual(r["bl"], Rect(0, 17, 50, 23))
        self.assertEqual(r["br"], Rect(51, 17, 49, 23))

    def test_hit_distinguishes_panes_and_dividers(self) -> None:
        g = GridLayout(fx_top=0.5, fx_bottom=0.7, fy=0.4)
        self.assertEqual(g.hit(50, 5, 100, 40), "vt")
        self.assertEqual(g.hit(70, 20, 100, 40), "vb")
        self.assertEqual(g.hit(5, 16, 100, 40), "h")
        self.assertEqual(g.hit(50, 16, 100, 40), "h")
        self.assertEqual(g.hit(0, 0, 100, 40), "tl")
        self.assertEqual(g.hit(99, 0, 100, 40), "tr")
        self.assertEqual(g.hit(0, 39, 100, 40), "bl")
        self.assertEqual(g.hit(60, 39, 100, 40), "bl")
        self.assertEqual(g.hit(99, 39, 100, 40), "br")

    def test_vertical_dividers_move_independently(self) -> None:
        g = GridLayout()
        g.drag("vt", 30, 5, 100, 40)
        g.drag("vb", 80, 30, 100, 40)
        self.assertEqual(g.splits(100, 40)[:2], (30, 80))
        r = g.rects(100, 40)
        self.assertEqual((r["tl"].w, r["bl"].w), (30, 80))

    def test_drag_moves_dividers_with_min_size_clamp(self) -> None:
        g = GridLayout()
        g.drag("vt", 2, 0, 100, 40)
        self.assertEqual(g.splits(100, 40)[0], GridLayout.MIN_W)
        self.assertEqual(g.splits(100, 40)[1], 50)  # bottom untouched
        g.drag("h", 0, 39, 100, 40)
        self.assertEqual(g.splits(100, 40)[2], 40 - 1 - GridLayout.MIN_H)

    def test_fractions_survive_resize(self) -> None:
        g = GridLayout()
        g.drag("vt", 70, 0, 100, 40)
        self.assertEqual(g.splits(200, 40)[0], 140)

    def test_tiny_terminal_yields_no_negative_rects(self) -> None:
        g = GridLayout(fx_top=0.9, fx_bottom=0.9, fy=0.9)
        for name, rect in g.rects(5, 3).items():
            self.assertGreaterEqual(rect.w, 0, name)
            self.assertGreaterEqual(rect.h, 0, name)


class RingBufferTests(unittest.TestCase):
    def test_wraparound_and_views(self) -> None:
        rb = RingBuffer(maxlen=5)
        for i in range(8):
            rb.append(f"l{i}", tag=i)
        self.assertEqual(len(rb), 5)
        self.assertEqual(rb.revision, 8)
        self.assertEqual(rb.view(0, 2), [(6, "l6"), (7, "l7")])
        self.assertEqual(rb.view(2, 2), [(4, "l4"), (5, "l5")])
        self.assertEqual(rb.view(100, 3), [])
        self.assertEqual(rb.all_text(), "l3\nl4\nl5\nl6\nl7")


class SanitizeTests(unittest.TestCase):
    def test_strips_escape_and_control_characters(self) -> None:
        # payloads: OSC window-title write, C1 CSI
        self.assertEqual(
            tui_core.sanitize("evil\x1b]0;pwned\x07 t\x9bmore\ttab"),
            "evil]0;pwned tmore tab",
        )


class TokenAtTests(unittest.TestCase):
    def test_url_hash_and_number(self) -> None:
        text = "fix 5144acb see https://ffmpeg.org/x (PR #123)"
        self.assertEqual(tui_core.token_at(text, text.index("144")), "5144acb")
        self.assertEqual(tui_core.token_at(text, text.index("org")),
                         "https://ffmpeg.org/x")
        self.assertEqual(tui_core.token_at(text, text.index("#123") + 1), "123")
        self.assertIsNone(tui_core.token_at(text, 0))
        self.assertIsNone(tui_core.token_at(text, text.index("PR")))

    def test_byline_author_and_branch_values_are_copyable(self) -> None:
        text = "author michaelni   branch ff-tmp-pgssubdec-4"
        self.assertEqual(tui_core.token_at(text, text.index("michaelni")),
                         "michaelni")
        self.assertEqual(tui_core.token_at(text, text.index("ff-tmp") + 3),
                         "ff-tmp-pgssubdec-4")
        self.assertIsNone(tui_core.token_at(text, text.index("branch")))


class RenderMarkdownTests(unittest.TestCase):
    def test_constructs(self) -> None:
        lines = render_markdown(
            "# Title\n"
            "Some **bold** and `code` and *italic* words.\n"
            "\n"
            "- first bullet that is long enough to wrap over the small width\n"
            "2. numbered\n"
            "> a quote\n"
            "```\n"
            "verbatim   line   kept-as-is beyond the width limit for sure\n"
            "```\n",
            width=30,
        )
        texts = [line_text(x) for x in lines]
        self.assertEqual(texts[0], "Title")
        self.assertEqual(lines[0][0][0], "h1")
        got = styles(lines)
        for style in ("bold", "code", "italic", "bullet", "quote", "codeblock"):
            self.assertIn(style, got)
        # hanging indent on the wrapped bullet continuation
        bullet_idx = next(i for i, x in enumerate(lines) if x and x[0][0] == "bullet")
        self.assertTrue(texts[bullet_idx + 1].startswith("  "))
        # fenced content is verbatim (inner spacing kept) and clipped
        fence = next(x for x in lines if x and x[0][0] == "codeblock")
        self.assertIn("verbatim   line", fence[0][1])
        self.assertLessEqual(len(fence[0][1]), 30)

    def test_gfm_extras(self) -> None:
        lines = render_markdown(
            "| Name | Qty |\n"
            "|:-----|----:|\n"
            "| foo | 1 |\n"
            "\n"
            "---\n"
            "- [x] done thing\n"
            "- [ ] open thing\n"
            "~~gone~~ [FFmpeg](https://ffmpeg.org) plain\n",
            width=40,
        )
        got = styles(lines)
        for s in ("th", "table_border", "hr", "checkbox_on", "checkbox_off",
                  "strike", "link", "url"):
            self.assertIn(s, got)
        texts = [line_text(x) for x in lines]
        self.assertTrue(all(len(t) <= 40 for t in texts))
        self.assertTrue(any(set(t) == {"─"} for t in texts))       # rule
        self.assertRegex(next(t for t in texts if "foo" in t),
                         r"foo\s* │ \s*1")                          # right-aligned
        self.assertIn("✔ done thing", texts)
        self.assertIn("☐ open thing", texts)

    def test_quote_paragraph_merges_and_gets_a_gutter(self) -> None:
        lines = render_markdown("> first part\n> second part\n", width=20)
        self.assertTrue(all(line[0] == ("quote_bar", "▌ ") for line in lines))
        self.assertGreater(len(lines), 1)  # merged text re-wrapped at width

    def test_fence_language_tag_and_block_width(self) -> None:
        lines = render_markdown("```c\nint x;\n```\n", width=20)
        self.assertEqual(lines[-1][0][0], "codeblock")
        self.assertEqual(len(lines[-1][0][1]), 20)  # full-width background
        self.assertIn([("codeblock_lang", " c")], lines)

    def test_emitted_styles_stay_in_the_documented_set(self) -> None:
        lines = render_markdown(
            "# h\n## h\n### h\n#### h\ntext **b** *i* ***bi*** `c` ~~s~~\n"
            "[l](http://u) http://bare\n> q\n- b\n1. n\n- [ ] t\n---\n"
            "|a|b|\n|-|-|\n|1|2|\n```py\nx\n```\n<!-- hidden -->\n",
            width=40,
        )
        self.assertLessEqual(styles(lines), tui_core.MARKDOWN_STYLES)

    def test_html_comment_block_shows_dimmed(self) -> None:
        lines = render_markdown(
            "LGTM overall.\n"
            "\n"
            "<!-- Scope gpt-5.6 code review: exhaustive over both commits,\n"
            "deep on the seek path.\n"
            "Scope combiner: verified the doxy claim. -->\n"
            "\n"
            "Please also update the docs.\n",
            width=40,
        )
        texts = [line_text(x) for x in lines]
        comment = [line_text(x) for x in lines if x and x[0][0] == "comment"]
        self.assertTrue(comment[0].startswith("<!--"))
        self.assertTrue(comment[-1].endswith("-->"))
        self.assertIn("Scope combiner: verified the doxy",
                      " ".join(comment))
        self.assertTrue(all(len(t) <= 40 for t in texts))
        for prose in ("LGTM overall.", "Please also update the docs."):
            row = next(x for x in lines if line_text(x) == prose)
            self.assertEqual(row[0][0], "text")

    def test_wrap_width_bound(self) -> None:
        lines = render_markdown("word " * 50, width=24)
        self.assertTrue(all(len(line_text(x)) <= 24 for x in lines))

    def test_a_forge_comment_renders(self) -> None:
        body = (
            "> The doxy says the index is in stream time base units,\n"
            "> which is not what the seek path assumes.\n"
            "\n"
            "Agreed -- `AVIndexEntry.timestamp` is the one to trust here,\n"
            "and `av_index_search_timestamp()` already does.\n"
        )
        lines = render_markdown(body, width=72)
        self.assertTrue(lines)
        got = styles(lines)
        self.assertIn("code", got)
        self.assertIn("quote", got)
        segments = [seg for line in lines for seg in line]
        self.assertTrue(all("`" not in t for s, t in segments if s == "code"))


class ChainTests(unittest.TestCase):
    def test_len_slice_and_iteration_span_the_parts(self) -> None:
        c = tui_core.Chain([[("a", "1")], [("a", "2")]], [],
                           [[("b", "3")], [("b", "4")]])
        self.assertEqual(len(c), 4)
        self.assertEqual(c[1:3], [[("a", "2")], [("b", "3")]])
        self.assertEqual(c[:], list(c))
        self.assertEqual(c[3:99], [[("b", "4")]])
        self.assertEqual(c[2:2], [])

    def test_sections_of_a_part_are_offset_into_the_chain(self) -> None:
        class Sectioned(list):
            """A part that locates its own headers, as
            diff_render.DiffView locates a diff's patches and hunks."""

            @staticmethod
            def sections(style: str) -> list[int]:
                return {"h2": [0, 2]}.get(style, [])

        inner = tui_core.Chain([[("a", "1")]],
                               Sectioned([[("b", str(i))] for i in range(3)]))
        self.assertEqual(inner.sections("h2"), [1, 3])
        self.assertEqual(inner.sections("h3"), [])
        self.assertEqual(tui_core.Chain([[("a", "0")]], inner).sections("h2"),
                         [2, 4])


class TileBlocksTests(unittest.TestCase):
    A = [[("text", "aaaa")], [("text", "a2")]]
    B = [[("text", "bb")]]
    C = [[("text", "cccc")], [("text", "c2")], [("text", "c3")]]

    @staticmethod
    def text(lines):
        return [line_text(line) for line in lines]

    def test_blocks_share_a_row_when_wide(self) -> None:
        # columns 4 and 2 wide plus the divider fit width 10; C wraps
        # under a full-grid-width rule line.
        got = self.text(tui_core.tile_blocks([self.A, self.B, self.C], 10))
        self.assertEqual(got, ["aaaa │ bb", "a2   │ ", "─────────",
                               "cccc", "c2", "c3"])

    def test_stacks_when_narrow(self) -> None:
        got = self.text(tui_core.tile_blocks([self.A, self.B], 5))
        self.assertEqual(got, ["aaaa", "a2", "────", "bb"])

    def test_each_column_is_as_wide_as_its_own_blocks(self) -> None:
        # A wide block must not widen the other columns: B's column
        # stays 2 cells, so both blocks fit width 13 side by side.
        wide = [[("text", "xxxxxxxx")]]
        got = self.text(tui_core.tile_blocks([self.B, wide], 13))
        self.assertEqual(got, ["bb │ xxxxxxxx"])

    def test_empty_input(self) -> None:
        self.assertEqual(tui_core.tile_blocks([], 80), [])
        self.assertEqual(tui_core.tile_blocks([[]], 80), [])



if __name__ == "__main__":
    unittest.main()
