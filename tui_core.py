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

Terminal-UI building blocks with no terminal dependency.

What belongs here: pure data/logic for fairy_tui.py -- the 2x2 grid
layout math, the block tiler, the scrollback ring buffer and the
markdown-to-styled-lines renderer. Everything is unit-testable
without a tty.

What does NOT belong: anything importing blessed or touching the
terminal, and anything review-specific (decisions, pipelines, forges).
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from itertools import islice
from threading import Lock

__all__ = ["Rect", "GridLayout", "RingBuffer", "StyledLine", "MARKDOWN_STYLES", "wrap",
           "render_markdown", "sanitize", "tile_blocks", "token_at"]

# (style, text) segments; the painter treats an unknown style as "text".
StyledLine = list[tuple[str, str]]

MARKDOWN_STYLES = frozenset({
    "h1", "h2", "h3", "h4", "bold", "italic", "bold_italic", "strike",
    "code", "codeblock", "codeblock_lang", "quote", "quote_bar", "bullet",
    "checkbox_on", "checkbox_off", "link", "url", "hr",
    "table_border", "th", "text",
})


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Strip C0/C1 control characters (tabs become one space) so
    forge/LLM-controlled text cannot inject terminal escape sequences
    into a pane."""
    return _CONTROL_RE.sub("", text.replace("\t", " "))


# Copyable things, most specific first: URLs, git hashes, #numbers,
# and the values behind the detail byline's author/branch labels.
_TOKEN_RES = (
    re.compile(r"https?://[^\s│()\[\]>\"']+"),
    re.compile(r"\b[0-9a-f]{7,40}\b"),
    re.compile(r"#\d+"),
    re.compile(r"(?<=\bauthor )[^\s│]+"),
    re.compile(r"(?<=\bbranch )[^\s│]+"),
)


def token_at(text: str, col: int) -> str | None:
    """The URL / git hash / issue-PR number covering column ``col`` of
    ``text``, else None. ``#123`` yields the bare number, ready for
    pasting into commands."""
    for rx in _TOKEN_RES:
        for m in rx.finditer(text):
            if m.start() <= col < m.end():
                return m.group().lstrip("#")
    return None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


class GridLayout:
    """2x2 pane grid: one full-width horizontal divider at fraction
    ``fy`` plus an independent vertical divider per half (``fx_top``,
    ``fx_bottom``), so the upper and lower column splits move separately.
    Fractions survive terminal resizes; pixel positions are derived per
    call."""

    MIN_W = 12
    MIN_H = 4

    def __init__(self, fx_top: float = 0.5, fx_bottom: float = 0.5,
                 fy: float = 0.4) -> None:
        self.fx_top = fx_top
        self.fx_bottom = fx_bottom
        self.fy = fy

    def _col(self, fx: float, w: int) -> int:
        col = _clamp(round(w * fx), self.MIN_W, max(self.MIN_W, w - 1 - self.MIN_W))
        return min(col, max(1, w - 2))

    def splits(self, w: int, h: int) -> tuple[int, int, int]:
        """(top divider column, bottom divider column, divider row),
        clamped to keep every pane at least MIN_W x MIN_H when the
        terminal allows it."""
        row = _clamp(round(h * self.fy), self.MIN_H, max(self.MIN_H, h - 1 - self.MIN_H))
        return self._col(self.fx_top, w), self._col(self.fx_bottom, w), min(row, max(1, h - 2))

    def rects(self, w: int, h: int) -> dict[str, Rect]:
        """Pane name ("tl" "tr" "bl" "br") -> Rect; the divider cells
        belong to no pane."""
        col_t, col_b, row = self.splits(w, h)
        bottom_h = max(0, h - row - 1)
        return {
            "tl": Rect(0, 0, col_t, row),
            "tr": Rect(col_t + 1, 0, max(0, w - col_t - 1), row),
            "bl": Rect(0, row + 1, col_b, bottom_h),
            "br": Rect(col_b + 1, row + 1, max(0, w - col_b - 1), bottom_h),
        }

    def hit(self, x: int, y: int, w: int, h: int) -> str:
        """Pane name under (x, y), or "vt"/"vb"/"h" for a divider cell."""
        col_t, col_b, row = self.splits(w, h)
        if y == row:
            return "h"
        if y < row:
            return "vt" if x == col_t else ("tl" if x < col_t else "tr")
        return "vb" if x == col_b else ("bl" if x < col_b else "br")

    def drag(self, grabbed: str, x: int, y: int, w: int, h: int) -> None:
        """Move the grabbed divider ("vt", "vb" or "h") toward (x, y)."""
        if w > 0 and grabbed in ("vt", "vb"):
            fx = _clamp(x, self.MIN_W, max(self.MIN_W, w - 1 - self.MIN_W)) / w
            setattr(self, "fx_top" if grabbed == "vt" else "fx_bottom", fx)
        if grabbed == "h" and h > 0:
            self.fy = _clamp(y, self.MIN_H, max(self.MIN_H, h - 1 - self.MIN_H)) / h


_TILE_SEP = " │ "


def tile_blocks(blocks: list[list[StyledLine]], width: int) -> list[StyledLine]:
    """Lay line blocks out side by side, as many columns as fit
    ``width``, divider lines between the columns and between block
    rows. Blocks fill rows left to right and each column is as wide as
    its own widest block, so one wide block does not stack everything."""
    blocks = [b for b in blocks if b]
    if not blocks:
        return []
    widths = [max(sum(len(t) for _, t in line) for line in b) or 1
              for b in blocks]
    for ncols in range(len(blocks), 0, -1):
        col_w = [max(widths[c::ncols]) for c in range(ncols)]
        if ncols == 1 or sum(col_w) + len(_TILE_SEP) * (ncols - 1) <= width:
            break
    grid_w = min(width, sum(col_w) + len(_TILE_SEP) * (len(col_w) - 1))
    out: list[StyledLine] = []
    for start in range(0, len(blocks), ncols):
        row = blocks[start:start + ncols]
        if out:
            out.append([("divider", "─" * grid_w)])
        for y in range(max(len(b) for b in row)):
            line: StyledLine = []
            for i, b in enumerate(row):
                cell = b[y] if y < len(b) else []
                # the last column keeps its natural width (no pad/clip)
                line += (cell if i == len(row) - 1
                         else _fit(cell, col_w[i]) + [("divider", _TILE_SEP)])
            out.append(line)
    return out


class RingBuffer:
    """Bounded scrollback of ``(tag, line)`` pairs shared between
    appender threads and a painter; the caller-chosen tag (e.g. a log
    level) lets the painter style lines. ``revision`` bumps on every
    append for cheap dirty checks."""

    def __init__(self, maxlen: int = 50_000) -> None:
        self._lines: deque[tuple[object, str]] = deque(maxlen=maxlen)
        self._lock = Lock()
        self.revision = 0

    def __len__(self) -> int:
        return len(self._lines)

    def append(self, line: str, tag: object = None) -> None:
        with self._lock:
            self._lines.append((tag, line))
            self.revision += 1

    def view(self, offset_from_end: int, count: int) -> list[tuple[object, str]]:
        """``count`` (tag, line) pairs ending ``offset_from_end`` lines
        above the newest; offsets beyond the start return what exists.
        Walks from the newest end so the follow-tail case (offset 0) is
        O(count), not O(buffer) -- this runs on every repaint."""
        with self._lock:
            n = len(self._lines)
            end = max(0, n - max(0, offset_from_end))
            start = max(0, end - count)
            return list(islice(reversed(self._lines), n - end, n - start))[::-1]

    def all_text(self) -> str:
        with self._lock:
            return "\n".join(line for _, line in self._lines)


_HEADING_RE = re.compile(r"(#{1,6})\s+(.*)")
_BULLET_RE = re.compile(r"(\s*)([-*+]|\d+[.)])\s+(.*)")
_CHECKBOX_RE = re.compile(r"\[( |x|X)\]\s+(.*)")
_HR_RE = re.compile(r"\s*([-*_])(\s*\1){2,}\s*$")
_QUOTE_RE = re.compile(r"\s*(?:>\s?)+(.*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_TABLE_SEP_RE = re.compile(r"\s*\|?[\s:|-]+\|?\s*$")
_INLINE_RE = re.compile(
    r"(\*\*\*.+?\*\*\*|\*\*.+?\*\*|\*[^*\s][^*]*\*|~~.+?~~|`[^`]+`"
    r"|\[[^\]]+\]\([^)\s]+\)|https?://[^\s)\]>]+)"
)


def _inline(text: str, base: str = "text") -> list[tuple[str, str]]:
    segs: list[tuple[str, str]] = []
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("***") and part.endswith("***") and len(part) > 6:
            segs.append(("bold_italic", part[3:-3]))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            segs.append(("bold", part[2:-2]))
        elif part.startswith("~~") and part.endswith("~~") and len(part) > 4:
            segs.append(("strike", part[2:-2]))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            segs.append(("code", part[1:-1]))
        elif (link := _LINK_RE.fullmatch(part)):
            label, url = link.groups()
            segs.append(("link", label))
            if url != label:
                segs.append(("url", f" ({url})"))
        elif part.startswith("http"):
            segs.append(("link", part))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            segs.append(("italic", part[1:-1]))
        else:
            segs.append((base, part))
    return segs


def _fit(segs: list[tuple[str, str]], target: int, align: str = "l") -> list[tuple[str, str]]:
    """Clip/pad styled segments to exactly ``target`` cells."""
    out: list[tuple[str, str]] = []
    used = 0
    for style, txt in segs:
        if used >= target:
            break
        txt = txt[:target - used]
        out.append((style, txt))
        used += len(txt)
    pad = target - used
    if pad <= 0:
        return out
    if align == "r":
        return [("text", " " * pad), *out]
    if align == "c":
        return [("text", " " * (pad // 2)), *out, ("text", " " * (pad - pad // 2))]
    return [*out, ("text", " " * pad)]


def _render_table(rows: list[str], width: int) -> list[StyledLine]:
    parsed = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    aligns: list[str] = []
    header: list[str] | None = None
    body = parsed
    if (len(parsed) >= 2 and "-" in rows[1] and _TABLE_SEP_RE.fullmatch(rows[1])):
        header, body = parsed[0], parsed[2:]
        aligns = ["c" if c.startswith(":") and c.endswith(":") else
                  "r" if c.endswith(":") else "l" for c in parsed[1]]
    ncols = max(len(r) for r in parsed)
    aligns += ["l"] * (ncols - len(aligns))
    cells = [
        [_inline(c, base="th" if r == header else "text")
         for c in r + [""] * (ncols - len(r))]
        for r in ([header] if header else []) + body
    ]
    widths = [max(sum(len(t) for _, t in row[i]) for row in cells)
              for i in range(ncols)]
    while sum(widths) + 3 * (ncols - 1) > width and max(widths) > 3:
        widths[widths.index(max(widths))] -= 1
    out: list[StyledLine] = []
    for r, row in enumerate(cells):
        line: StyledLine = []
        for i, cell in enumerate(row):
            if i:
                line.append(("table_border", " │ "))
            line += _fit(cell, widths[i], aligns[i])
        out.append(line)
        if header and r == 0:
            out.append([("table_border", "─┼─".join("─" * w for w in widths))])
    return out


def _atoms(segs: list[tuple[str, str]]) -> list[StyledLine]:
    """Whitespace-delimited atoms; an atom spans segment boundaries when
    no space separates them, so ``**bold**,`` keeps its comma attached."""
    atoms: list[StyledLine] = []
    open_atom = False
    for style, txt in segs:
        for part in re.split(r"(\s+)", txt):
            if not part:
                continue
            if part.isspace():
                open_atom = False
            elif open_atom:
                atoms[-1].append((style, part))
            else:
                atoms.append([(style, part)])
                open_atom = True
    return atoms


def wrap(
    segs: list[tuple[str, str]],
    width: int,
    initial: tuple[str, str] = ("text", ""),
    subsequent: str = "",
) -> list[StyledLine]:
    """Greedy wrap of styled segments; ``initial`` is a styled prefix
    for the first line (e.g. a bullet marker), ``subsequent`` the
    hanging indent for the rest."""
    atoms = _atoms(segs)
    if not atoms:
        return []
    lines: list[StyledLine] = []
    prefix: tuple[str, str] = initial
    cur: StyledLine = []
    cur_len = 0

    def emit() -> StyledLine:
        return [prefix, *cur] if prefix[1] else list(cur)

    for atom in atoms:
        alen = sum(len(t) for _, t in atom)
        extra = alen + (1 if cur else 0)
        if cur and len(prefix[1]) + cur_len + extra > width:
            lines.append(emit())
            prefix = ("text", subsequent)
            cur = list(atom)
            cur_len = alen
        else:
            if cur:
                # The joining space inherits the next atom's style so
                # underline/strike runs stay continuous inside a span.
                cur.append((atom[0][0], " "))
            cur.extend(atom)
            cur_len += extra
    lines.append(emit())
    return lines


def render_markdown(text: str, width: int) -> list[StyledLine]:
    """Render a markdown message to width-bounded styled lines: ATX
    headings, bold/italic/strike/code/link spans, fenced blocks (verbatim,
    full-width for a background, language tag kept), pipe tables with
    alignment, ``-``/``*``/``1.`` lists with hanging indent and
    checkboxes, gutter-barred re-wrapped quotes, horizontal rules and
    wrapped paragraphs."""
    width = max(8, width)
    out: list[StyledLine] = []
    para: list[str] = []
    quote: list[str] = []
    table: list[str] = []
    in_fence = False

    def flush() -> None:
        if para:
            out.extend(wrap(_inline(" ".join(para)), width))
            para.clear()
        if quote:
            for ln in wrap(_inline(" ".join(quote), base="quote"), width - 2):
                out.append([("quote_bar", "▌ "), *ln])
            quote.clear()
        if table:
            out.extend(_render_table(table, width))
            table.clear()

    def blank() -> None:
        if out and out[-1]:
            out.append([])

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            if not in_fence:
                blank()
                if (lang := stripped[3:].strip()):
                    out.append([("codeblock_lang", f" {lang}"[:width])])
            in_fence = not in_fence
            continue
        if in_fence:
            # Full-width so the painter's background reads as a block.
            out.append([("codeblock", (" " + raw).ljust(width)[:width])])
            continue
        if stripped.startswith("|"):
            if not table:
                flush()
            table.append(line)
            continue
        if table:
            flush()
        if not stripped:
            flush()
            blank()
            continue
        if (heading := _HEADING_RE.match(line)):
            flush()
            blank()
            style = f"h{min(len(heading.group(1)), 4)}"
            out.extend(wrap([(style, heading.group(2))], width))
            continue
        if _HR_RE.fullmatch(line):
            flush()
            out.append([("hr", "─" * width)])
            continue
        if (quoted := _QUOTE_RE.fullmatch(line)):
            if para:
                flush()
            quote.append(quoted.group(1))
            continue
        if quote:
            flush()
        if (bullet := _BULLET_RE.match(line)):
            flush()
            indent, marker, rest = bullet.groups()
            if (box := _CHECKBOX_RE.match(rest)):
                done = box.group(1).lower() == "x"
                out.extend(wrap(
                    _inline(box.group(2)), width,
                    initial=("checkbox_on" if done else "checkbox_off",
                             f"{indent}{'✔' if done else '☐'} "),
                    subsequent=" " * (len(indent) + 2),
                ))
                continue
            marker_out = "• " if marker in "-*+" else f"{marker} "
            out.extend(wrap(
                _inline(rest), width,
                initial=("bullet", f"{indent}{marker_out}"),
                subsequent=" " * (len(indent) + len(marker_out)),
            ))
            continue
        para.append(stripped)
    flush()
    while out and not out[-1]:
        out.pop()
    return out
