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

import logging
import re
from typing import Callable

__all__ = ["hide_scope_block", "link_published_branch"]

logger = logging.getLogger(__name__)


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


def _link_mentions(message: str, pattern: str,
                   url_of: Callable[[str], str | None]) -> str:
    """Turn every bare or backticked match of ``pattern`` that
    ``url_of`` knows a page for into a markdown link; matches already
    inside a link or URL stay."""
    def link(match: re.Match[str]) -> str:
        text = match.group(2)
        url = url_of(text)
        return match.group(0) if url is None else f"[{text}]({url})"
    return re.sub(rf"(?<![\w/\[`+-])(`?)({pattern})\1(?![\w/\]`+-])", link, message)


def link_published_branch(message: str, forge_branch: str, sha: str,
                          branch_url: str, commit_url: str) -> str:
    """Link the mentions of a published branch (its forge name, e.g.
    ``fairy/<name>``) and of its tip commit (any abbreviation of
    ``sha``) to their forge pages; fenced code blocks stay verbatim."""
    def link(prose: str) -> str:
        prose = _link_mentions(prose, re.escape(forge_branch), lambda _: branch_url)
        return _link_mentions(prose, "[0-9a-f]{7,}",
                              lambda text: commit_url if sha.startswith(text) else None)
    parts = re.split(r"(```.*?```)", message, flags=re.DOTALL)
    return "".join(part if i % 2 else link(part) for i, part in enumerate(parts))
