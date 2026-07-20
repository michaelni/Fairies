"""tui_core: grid layout math, ring buffer, markdown renderer."""

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

    def test_wrap_width_bound(self) -> None:
        lines = render_markdown("word " * 50, width=24)
        self.assertTrue(all(len(line_text(x)) <= 24 for x in lines))

    def test_real_issue_comment_renders(self) -> None:
        comments = json.loads(
            (REPO_ROOT / "tests/fixtures/issue_fairy/ffmpeg_issue_23738_comments.json")
            .read_text()
        )
        body = comments[0]["body"]
        lines = render_markdown(body, width=72)
        self.assertTrue(lines)
        got = styles(lines)
        self.assertIn("code", got)
        self.assertIn("quote", got)
        segments = [seg for line in lines for seg in line]
        self.assertTrue(all("`" not in t for s, t in segments if s == "code"))


if __name__ == "__main__":
    unittest.main()
