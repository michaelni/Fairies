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
markdown and git-diff renderers to styled lines. Everything is
unit-testable without a tty.

What does NOT belong: anything importing blessed or touching the
terminal, and anything review-specific (decisions, pipelines, forges).
"""

from __future__ import annotations

import difflib
import re
from collections import deque
from dataclasses import dataclass
from itertools import accumulate, groupby, islice
from threading import Lock

from pygments.lexer import Lexer
from pygments.lexers import get_lexer_for_filename
from pygments.token import Comment, Keyword, Name, Number, Operator, String
from pygments.util import ClassNotFound

__all__ = ["Rect", "GridLayout", "RingBuffer", "StyledLine", "MARKDOWN_STYLES",
           "DIFF_STYLES", "wrap", "render_markdown", "render_diff", "sanitize",
           "tile_blocks", "token_at"]

# (style, text) segments; the painter treats an unknown style as "text".
StyledLine = list[tuple[str, str]]

MARKDOWN_STYLES = frozenset({
    "h1", "h2", "h3", "h4", "bold", "italic", "bold_italic", "strike",
    "code", "codeblock", "codeblock_lang", "quote", "quote_bar", "bullet",
    "checkbox_on", "checkbox_off", "link", "url", "hr",
    "table_border", "th", "text", "comment",
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
    checkboxes, gutter-barred re-wrapped quotes, horizontal rules,
    wrapped paragraphs, and HTML comments -- which forges hide but a
    review pane must show, dimmed, delimiters kept."""
    width = max(8, width)
    out: list[StyledLine] = []
    para: list[str] = []
    quote: list[str] = []
    table: list[str] = []
    in_fence = False
    in_comment = False

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
        if not in_fence and (in_comment or stripped.startswith("<!--")):
            flush()
            if stripped:
                out.extend(wrap([("comment", stripped)], width))
            else:
                blank()
            in_comment = "-->" not in stripped
            continue
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


DIFF_STYLES = frozenset(
    {f"df_{bg}_{fg}" for bg in ("ctx", "add", "del", "addhl", "delhl")
     for fg in ("tx", "kw", "ty", "fn", "str", "com", "num")}
    | {"diff_file", "diff_hunk", "diff_meta", "diff_commit",
       "text", "bold", "sc_good", "sc_bad"})

# Most specific first: a token maps to the first ancestor listed, so
# Comment.Preproc (C's #include/#define) must outrank Comment.
_DIFF_TOKEN_FG = {
    Comment.PreprocFile: "str", Comment.Preproc: "kw", Comment: "com",
    String: "str", Number: "num",
    Keyword.Type: "ty", Keyword: "kw", Operator.Word: "kw",
    Name.Function: "fn", Name.Decorator: "fn", Name.Variable: "fn",
    Name.Label: "fn", Name.Class: "ty", Name.Builtin: "kw",
}

_HUNK_RE = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_MBOX_FROM_RE = re.compile(r"From [0-9a-f]{40} ")
_DIFFSTAT_RE = re.compile(r"( \S.*\| +\d+ ?)(\+*)(-*)")
_DIFF_WORD_RE = re.compile(r"\w+|\s+|[^\w\s]+")


def _lexer_for(path: str, cache: dict[str, Lexer | None]) -> Lexer | None:
    name = path.rsplit("/", 1)[-1]
    key = name[name.rfind("."):] if "." in name else name
    if key not in cache:
        try:
            # stripnl would drop leading/trailing blank lines and shift
            # _hunk_fgs' row mapping against the hunk's lines.
            cache[key] = get_lexer_for_filename(key, stripnl=False)
        except ClassNotFound:
            cache[key] = None
    return cache[key]


def _changed_spans(
    a: str, b: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Word-level change spans (char ranges) of a paired -/+ line;
    ([], []) when the lines share too little for marks to help."""
    ta, tb = _DIFF_WORD_RE.findall(a), _DIFF_WORD_RE.findall(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks())
    if matched < max(1, min(len(ta), len(tb)) / 2):
        return [], []
    pa = list(accumulate(map(len, ta), initial=0))
    pb = list(accumulate(map(len, tb), initial=0))
    ops = sm.get_opcodes()
    return ([(pa[i1], pa[i2]) for tag, i1, i2, _, _ in ops
             if tag in ("replace", "delete") and i1 < i2],
            [(pb[j1], pb[j2]) for tag, _, _, j1, j2 in ops
             if tag in ("replace", "insert") and j1 < j2])


def _hunk_fgs(lexer: Lexer | None, side: list[str]) -> list[list[str]]:
    """Per-character syntax classes for one side of a hunk. The side is
    lexed as a single text so a construct spanning lines (a block
    comment, a multi-line string) keeps its class on every line."""
    fgs = [["tx"] * len(code) for code in side]
    if lexer is None:
        return fgs
    row = col = 0
    for token, text in lexer.get_tokens("\n".join(side)):
        cls = next((c for t, c in _DIFF_TOKEN_FG.items() if token in t), "tx")
        for j, part in enumerate(text.split("\n")):
            if j:
                row, col = row + 1, 0
            if row >= len(fgs):
                return fgs
            end = min(col + len(part), len(fgs[row]))
            fgs[row][col:end] = [cls] * (end - col)
            col = end
    return fgs


def _code_line(prefix: str, code: str, bg: str, hl: list[tuple[int, int]],
               fg: list[str]) -> StyledLine:
    marked = [False] * len(code)
    for lo, hi in hl:
        marked[lo:hi] = [True] * (hi - lo)
    line: StyledLine = [(f"df_{bg}_tx", prefix)]
    for style, run in groupby(
            range(len(code)),
            lambda i: f"df_{bg + 'hl' if marked[i] else bg}_{fg[i]}"):
        run = list(run)
        line.append((style, code[run[0]:run[-1] + 1]))
    return line


def _emit_run(out: list[StyledLine], minus: list[tuple[str, list[str]]],
              plus: list[tuple[str, list[str]]]) -> None:
    spans = [_changed_spans(a, b) for (a, _), (b, _) in zip(minus, plus)]
    for i, (code, fg) in enumerate(minus):
        out.append(_code_line("-", code, "del",
                              spans[i][0] if i < len(spans) else [], fg))
    for i, (code, fg) in enumerate(plus):
        out.append(_code_line("+", code, "add",
                              spans[i][1] if i < len(spans) else [], fg))


def render_diff(patch: str) -> list[StyledLine]:
    """Render ``git diff`` / ``git format-patch --stdout`` text to styled
    lines: commit, file and hunk headers, colored diffstat, and code with
    added/removed line backgrounds, brighter word-level change marks and
    per-file syntax coloring. Tabs are expanded; lines are emitted
    unwrapped -- the painter clips them to the pane."""
    out: list[StyledLine] = []
    lexer: Lexer | None = None
    lexer_cache: dict[str, Lexer | None] = {}
    lines = patch.split("\n")
    in_mail_header = False
    in_commit_msg = False
    i = 0
    while i < len(lines):
        line = lines[i].expandtabs()
        i += 1
        if in_mail_header:
            in_mail_header = bool(line)
            in_commit_msg = not in_mail_header
            out.append([("bold" if line.startswith("Subject:")
                         else "diff_meta", line)] if line else [])
            continue
        if in_commit_msg:
            # Message text runs to the "---" scissors line; a quoted
            # hunk or header inside it must stay verbatim text.
            if line == "---":
                in_commit_msg = False
                out.append([("diff_meta", line)])
            else:
                out.append([("text", line)] if line else [])
            continue
        if (hunk := _HUNK_RE.match(line)):
            out.append([("diff_hunk", line)])
            rem_a, rem_b = int(hunk.group(1) or 1), int(hunk.group(2) or 1)
            body: list[tuple[str, str]] = []
            while (rem_a > 0 or rem_b > 0) and i < len(lines):
                raw = lines[i].expandtabs()
                i += 1
                op, code = raw[:1], raw[1:]
                body.append((op, code))
                if op == "-":
                    rem_a -= 1
                elif op == "+":
                    rem_b -= 1
                elif op != "\\":
                    rem_a -= 1
                    rem_b -= 1
            old = _hunk_fgs(lexer, [c for op, c in body
                                    if op not in ("+", "\\")])
            new = _hunk_fgs(lexer, [c for op, c in body
                                    if op not in ("-", "\\")])
            oi = ni = 0
            minus: list[tuple[str, list[str]]] = []
            plus: list[tuple[str, list[str]]] = []
            for op, code in body:
                if op == "-":
                    minus.append((code, old[oi]))
                    oi += 1
                elif op == "+":
                    plus.append((code, new[ni]))
                    ni += 1
                elif op == "\\":
                    _emit_run(out, minus, plus)
                    minus, plus = [], []
                    out.append([("diff_meta", op + code)])
                else:
                    _emit_run(out, minus, plus)
                    minus, plus = [], []
                    out.append(_code_line(" ", code, "ctx", [], new[ni]))
                    oi += 1
                    ni += 1
            _emit_run(out, minus, plus)
            continue
        if line.startswith("diff --git "):
            lexer = _lexer_for(line.rsplit(" b/", 1)[-1], lexer_cache)
            out.append([("diff_file", line)])
            continue
        if _MBOX_FROM_RE.match(line):
            in_mail_header = True
            out.append([("diff_commit", line)])
            continue
        if line == "---" or line.startswith(
                ("index ", "--- ", "+++ ", "old mode", "new mode", "new file",
                 "deleted file", "similarity index", "dissimilarity",
                 "rename from", "rename to", "copy from", "copy to",
                 "Binary files")):
            out.append([("diff_meta", line)])
            continue
        if (stat := _DIFFSTAT_RE.fullmatch(line)):
            out.append([(style, text) for style, text in
                        zip(("text", "sc_good", "sc_bad"), stat.groups())
                        if text])
            continue
        out.append([("text", line)] if line else [])
    while out and not out[-1]:
        out.pop()
    return out
