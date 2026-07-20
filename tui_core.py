"""Terminal-UI building blocks with no terminal dependency.

What belongs here: pure data/logic for fairy_tui.py -- the 2x2 grid
layout math, the scrollback ring buffer and the markdown-to-styled-lines
renderer. Everything is unit-testable without a tty.

What does NOT belong: anything importing blessed or touching the
terminal, and anything review-specific (decisions, pipelines, forges).
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from itertools import islice
from threading import Lock

__all__ = ["Rect", "GridLayout", "RingBuffer", "StyledLine", "render_markdown", "sanitize"]

# One rendered line: (style, text) segments. Styles come from the closed
# set emitted by render_markdown ("h1" "h2" "h3" "bold" "italic" "code"
# "codeblock" "quote" "bullet" "text"); the painter maps them to terminal
# attributes and treats unknown styles as "text".
StyledLine = list[tuple[str, str]]


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Strip C0/C1 control characters (tabs become one space) so
    forge/LLM-controlled text cannot inject terminal escape sequences
    into a pane."""
    return _CONTROL_RE.sub("", text.replace("\t", " "))


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


class RingBuffer:
    """Bounded line scrollback shared between appender threads and a
    painter; ``revision`` bumps on every append for cheap dirty checks."""

    def __init__(self, maxlen: int = 50_000) -> None:
        self._lines: deque[str] = deque(maxlen=maxlen)
        self._lock = Lock()
        self.revision = 0

    def __len__(self) -> int:
        return len(self._lines)

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            self.revision += 1

    def view(self, offset_from_end: int, count: int) -> list[str]:
        """``count`` lines ending ``offset_from_end`` lines above the
        newest; offsets beyond the start return what exists."""
        with self._lock:
            end = max(0, len(self._lines) - max(0, offset_from_end))
            start = max(0, end - count)
            return list(islice(self._lines, start, end))

    def all_text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)


_HEADING_RE = re.compile(r"(#{1,6})\s+(.*)")
_BULLET_RE = re.compile(r"(\s*)([-*+]|\d+[.)])\s+(.*)")
_INLINE_RE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*\*|`[^`]+`)")


def _inline(text: str, base: str = "text") -> list[tuple[str, str]]:
    segs: list[tuple[str, str]] = []
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            segs.append(("bold", part[2:-2]))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            segs.append(("code", part[1:-1]))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            segs.append(("italic", part[1:-1]))
        else:
            segs.append((base, part))
    return segs


def _wrap(
    segs: list[tuple[str, str]],
    width: int,
    initial: tuple[str, str] = ("text", ""),
    subsequent: str = "",
) -> list[StyledLine]:
    """Greedy word wrap of styled segments; ``initial`` is a styled
    prefix for the first line (e.g. a bullet marker), ``subsequent``
    the hanging indent for the rest."""
    words: list[tuple[str, str]] = [
        (style, word) for style, txt in segs for word in txt.split()
    ]
    if not words:
        return []
    lines: list[StyledLine] = []
    prefix: tuple[str, str] = initial
    cur: StyledLine = []
    cur_len = 0
    def emit() -> StyledLine:
        return [prefix, *cur] if prefix[1] else list(cur)

    for style, word in words:
        extra = len(word) + (1 if cur else 0)
        if cur and len(prefix[1]) + cur_len + extra > width:
            lines.append(emit())
            prefix = ("text", subsequent)
            cur = [(style, word)]
            cur_len = len(word)
        else:
            cur.append((style, (" " if cur else "") + word))
            cur_len += extra
    lines.append(emit())
    return lines


def render_markdown(text: str, width: int) -> list[StyledLine]:
    """Render a markdown message to width-bounded styled lines: ATX
    headings, ``**bold**``/``*italic*``/`` `code` `` spans, fenced blocks
    (verbatim, clipped not wrapped), ``-``/``*``/``1.`` lists with
    hanging indent, ``>`` quotes, wrapped paragraphs."""
    width = max(8, width)
    out: list[StyledLine] = []
    para: list[str] = []
    in_fence = False

    def flush_para() -> None:
        if para:
            out.extend(_wrap(_inline(" ".join(para)), width))
            para.clear()

    def blank() -> None:
        if out and out[-1]:
            out.append([])

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("```"):
            flush_para()
            if not in_fence:
                blank()
            in_fence = not in_fence
            continue
        if in_fence:
            out.append([("codeblock", raw[:width])])
            continue
        if not line.strip():
            flush_para()
            blank()
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush_para()
            blank()
            style = f"h{min(len(heading.group(1)), 3)}"
            out.extend(_wrap([(style, heading.group(2))], width))
            continue
        if line.lstrip().startswith(">"):
            flush_para()
            quoted = line.lstrip().lstrip(">").strip()
            out.extend(_wrap(
                _inline(quoted, base="quote"), width,
                initial=("quote", "> "), subsequent="> ",
            ))
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            flush_para()
            indent, marker, rest = bullet.groups()
            out.extend(_wrap(
                _inline(rest), width,
                initial=("bullet", f"{indent}{marker} "),
                subsequent=" " * (len(indent) + len(marker) + 1),
            ))
            continue
        para.append(line.strip())
    flush_para()
    while out and not out[-1]:
        out.pop()
    return out
