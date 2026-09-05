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

__all__ = ["hide_scope_block"]

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
