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

Workarounds for formatting lapses of LLM output, applied to a review
message after the model produced it.

What belongs here: small text fixes, one per observed lapse, that do
not change what a review says.

What does NOT belong: provider protocol handling and anything that
alters a review's content or classification.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Callable

__all__ = ["fold_entries", "hide_scope_block", "link_published_branch",
           "names_published_branch"]

logger = logging.getLogger(__name__)

_FENCED_CODE = re.compile(r"(```.*?```)", re.DOTALL)


def hide_scope_block(message: str) -> str:
    """Wrap a run of plain-text ``Scope ...`` paragraphs in the HTML
    comment the prompt asks for. A message without such a run, or with
    the run already inside a comment, is returned unchanged."""
    lines = message.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith("Scope ")), None)
    if start is None:
        return message
    before = "\n".join(lines[:start])
    if before.count("<!--") > before.count("-->"):
        return message
    end = start
    while end < len(lines) and (lines[end].startswith("Scope ") or not lines[end].strip()):
        end += 1
    while not lines[end - 1].strip():
        end -= 1
    scope = [line for line in lines[start:end] if line.strip()]
    logger.info("scope block hidden in an HTML comment (%d lines)", len(scope))
    return "\n".join(lines[:start] + ["<!--", *scope, "-->"] + lines[end:])


def _code_span(ticks: str) -> str:
    """Pattern of a markdown code span within a line, its backtick
    string captured as the group named ``ticks``."""
    end = rf"(?<!`)(?P={ticks})(?!`)"
    return rf"(?<!`)(?P<{ticks}>`+)(?!`)(?:(?!{end}).)+{end}"


_BOLD_ENTRY = re.compile(r"^(?P<marker>(?:\d+\.|[-*]) +)(?P<first>\*\*.+)$")
_BOLD_LEAD = re.compile(r"^(?P<lead>\*\*(?:(?!\*\*).)+\*\*[.:]?) +(?P<rest>\S.*)$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\*)([^*]+?)\*(?![*\w])")


def _inline_html(markdown: str) -> str:
    """One line of markdown as HTML: code spans, links, bold and
    italics converted, everything else escaped."""
    spans: list[str] = []

    def hold(match: re.Match[str]) -> str:
        spans.append(f"<code>{html.escape(match[0].strip('`').strip())}</code>")
        return f"\0{len(spans) - 1}\0"
    text = html.escape(re.sub(_code_span("ticks"), hold, markdown), quote=False)
    text = _LINK.sub(r'<a href="\2">\1</a>', text)
    text = _ITALIC.sub(r"<i>\1</i>", _BOLD.sub(r"<b>\1</b>", text))
    return re.sub("\0(\\d+)\0", lambda match: spans[int(match[1])], text)


def _fenced_block_end(lines: list[str], i: int) -> int:
    """Index past the fenced code block that opens at ``lines[i]``."""
    return next((j for j in range(i + 1, len(lines)) if lines[j].startswith("```")),
                len(lines) - 1) + 1


def _details_block(entry: re.Match[str], body: list[str]) -> list[str] | None:
    """The lines of a ``_BOLD_ENTRY`` match and its ``body`` as a
    ``<details>`` block, or None for an entry with nothing to fold."""
    indent = " " * len(entry["marker"])
    first = entry["first"]
    if not body:
        lead = _BOLD_LEAD.match(first)
        if lead is None:
            return None
        first, body = lead["lead"], [indent + lead["rest"]]
    return [f"{entry['marker']}<details><summary>{_inline_html(first)}</summary>",
            "", *body, f"{indent}</details>"]


def fold_entries(message: str) -> str:
    """Fold every list entry that opens with bold text into a
    ``<details>`` block: the opening line becomes the ``<summary>``, as
    HTML because the forge renders no markdown inside one, and the
    indented rest of the entry the folded body. An entry written as a
    single line is split after its bold lead. Fenced code blocks and
    entries with nothing to fold stay as they are."""
    lines = message.split("\n")
    out: list[str] = []
    folded = 0
    i = 0
    while i < len(lines):
        entry = _BOLD_ENTRY.match(lines[i])
        end = i + 1
        if lines[i].startswith("```"):
            end = _fenced_block_end(lines, i)
        elif entry:
            while end < len(lines) and (not lines[end].strip() or lines[end].startswith("  ")):
                end += 1
            while not lines[end - 1].strip():
                end -= 1
        block = _details_block(entry, lines[i + 1:end]) if entry else None
        out += block or lines[i:end]
        folded += block is not None
        i = end
    if folded:
        logger.info("%d review entries folded into <details> blocks", folded)
    return "\n".join(out)


def _link_mentions(message: str, pattern: str,
                   url_of: Callable[[str], str | None]) -> str:
    """Turn every bare or backticked match of ``pattern`` that
    ``url_of`` knows a page for, and every code span holding such a
    match, into a markdown link; matches already inside a link or URL
    stay."""
    mention = rf"(?<![\w/\[`+-])(?P<tick>`?)(?P<mention>{pattern})(?P=tick)(?![\w/\]`+-])"

    def link(match: re.Match[str]) -> str:
        mentions = [match] if match["mention"] else re.finditer(
            mention, (match["code"] or "").strip("`"))
        url = next(filter(None, (url_of(found["mention"]) for found in mentions)), None)
        return match[0] if url is None else f"[{match['mention'] or match[0]}]({url})"
    link_text = rf"(?:{_code_span('text_ticks')}|(?!{_code_span('lone_ticks')})[^\[\]\n])*"
    return re.sub(rf"\[{link_text}\]\([^)\s]*\)|{mention}|(?P<code>{_code_span('ticks')})",
                  link, message)


def _by_forge_name(prose: str, forge_branch: str) -> str:
    """``prose`` with a backticked ``<name>`` spelled as its forge
    branch ``forge_branch``, a ``<prefix>/<name>``."""
    return prose.replace(f"`{forge_branch.rpartition('/')[2]}`", f"`{forge_branch}`")


def link_published_branch(message: str, forge_branch: str, sha: str,
                          branch_url: str, commit_url: str) -> str:
    """Link the mentions of a published branch (its forge name, e.g.
    ``fairy/<name>``, or a backticked ``<name>``, linked under the
    forge name) and of its tip commit (any abbreviation of
    ``sha``) to their forge pages. A code span holding a mention is
    linked as a whole; fenced code blocks stay verbatim."""
    def link(prose: str) -> str:
        prose = _link_mentions(_by_forge_name(prose, forge_branch),
                               re.escape(forge_branch), lambda _: branch_url)
        return _link_mentions(prose, "[0-9a-f]{7,}",
                              lambda text: commit_url if sha.startswith(text) else None)
    parts = re.split(_FENCED_CODE, message)
    return "".join(part if i % 2 else link(part) for i, part in enumerate(parts))


def names_published_branch(message: str, forge_branch: str) -> bool:
    """Whether the message names the branch (its forge name, e.g.
    ``fairy/<name>``, bare, backticked or as link text, or a backticked
    ``<name>``) outside its fenced code blocks."""
    return any(re.search(rf"(?<![\w/+-]){re.escape(forge_branch)}(?![\w/+-])",
                         _by_forge_name(prose, forge_branch))
               for prose in re.split(_FENCED_CODE, message)[::2])
