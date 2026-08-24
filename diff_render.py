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

Git diff / format-patch rendering to styled terminal lines.

What belongs here: turning ``git diff`` / ``git format-patch
--stdout`` text into tui_core.StyledLine rows -- commit, file and
hunk header styling, the colored diffstat, add/del line backgrounds,
word-level change marks and per-file syntax coloring via pygments.

What does NOT belong: running git (git_util), terminal painting and
key handling (fairy_tui), markdown rendering and the other TUI
building blocks (tui_core).
"""

from __future__ import annotations

import difflib
import re
from itertools import accumulate, groupby

from pygments.lexer import Lexer
from pygments.lexers import get_lexer_for_filename
from pygments.token import Comment, Keyword, Name, Number, Operator, String
from pygments.util import ClassNotFound

from tui_core import StyledLine

__all__ = ["DIFF_FGS", "DIFF_STYLES", "DiffView", "render_diff"]

DIFF_FGS = ("tx", "kw", "ty", "fn", "str", "com", "num")

DIFF_STYLES = frozenset(
    {f"df_{bg}_{fg}" for bg in ("ctx", "add", "del", "addhl", "delhl")
     for fg in DIFF_FGS}
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
_token_fg_memo: dict = {}

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
        if (cls := _token_fg_memo.get(token)) is None:
            _token_fg_memo[token] = cls = next(
                (c for t, c in _DIFF_TOKEN_FG.items() if token in t), "tx")
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


def _render_hunk(body: list[tuple[str, str]],
                 lexer: Lexer | None) -> list[StyledLine]:
    """One styled line per ``(op, code)`` body entry, with the run
    grouping and word marks of a rendered hunk."""
    out: list[StyledLine] = []
    old = _hunk_fgs(lexer, [c for op, c in body if op not in ("+", "\\")])
    new = _hunk_fgs(lexer, [c for op, c in body if op not in ("-", "\\")])
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
    return out


class DiffView:
    """Lazily rendered ``git diff`` / ``git format-patch --stdout``
    text: commit, file and hunk headers, colored diffstat, and code
    with added/removed line backgrounds, brighter word-level change
    marks and per-file syntax coloring. Tabs are expanded; lines come
    out unwrapped -- the painter clips them to the pane.

    Construction runs only the cheap structural pass; the pygments
    work for a hunk happens the first time a slice covers one of its
    lines, and stays rendered. The intended cost is per visible line,
    but rendering is per hunk -- cross-line lexing and the word marks
    need a whole hunk -- so a diff that is one huge hunk (a new file,
    a whole-file rewrite) still pays its full highlight on first
    touch. Supports ``len``, unit-step slicing and iteration;
    iterating renders everything."""

    def __init__(self, patch: str) -> None:
        out: list[StyledLine | None] = []
        self._hunks: list[tuple[int, list[tuple[str, str]],
                                Lexer | None]] = []
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
                self._hunks.append((len(out), body, lexer))
                out += [None] * len(body)
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
                    ("index ", "--- ", "+++ ", "old mode", "new mode",
                     "new file", "deleted file", "similarity index",
                     "dissimilarity", "rename from", "rename to",
                     "copy from", "copy to", "Binary files")):
                out.append([("diff_meta", line)])
                continue
            if (stat := _DIFFSTAT_RE.fullmatch(line)):
                out.append([(style, text) for style, text in
                            zip(("text", "sc_good", "sc_bad"), stat.groups())
                            if text])
                continue
            out.append([("text", line)] if line else [])
        while out and out[-1] == []:
            out.pop()
        self._lines = out

    def __len__(self) -> int:
        return len(self._lines)

    def __iter__(self):
        self._materialize(0, len(self._lines))
        return iter(self._lines)

    def __getitem__(self, key: slice) -> list[StyledLine]:
        lo, hi, _ = key.indices(len(self._lines))
        self._materialize(lo, hi)
        return self._lines[key]

    def _materialize(self, lo: int, hi: int) -> None:
        for start, body, lexer in self._hunks:
            if start < hi and lo < start + len(body) \
                    and self._lines[start] is None:
                self._lines[start:start + len(body)] = \
                    _render_hunk(body, lexer)


def render_diff(patch: str) -> list[StyledLine]:
    """``DiffView(patch)`` rendered in full, as a plain line list."""
    return list(DiffView(patch))
