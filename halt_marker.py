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

The operator's halt marker: a file whose existence stops fairy -- the
agent's scanning and posting, the workers' claims and the reviews in
flight -- until the operator removes it. Its text is the reason.

What belongs here: where the marker lives, writing it, reading it.
What does NOT belong: reacting to it (agent, worker, pr_review_wrapper).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["path", "halt", "reason"]


def path(db_root: Path, configured: Path | None = None) -> Path:
    """--halt-file when configured, else ``halted`` in the db root."""
    return Path(configured) if configured else Path(db_root) / "halted"


def halt(marker: Path, reason: str) -> None:
    marker.write_text(reason, encoding="utf-8")


def reason(marker: Path) -> str | None:
    """The marker's text; None while fairy is not halted."""
    try:
        return marker.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
