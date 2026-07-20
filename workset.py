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

Persistent per-item work files for the review pipeline.

One JSON file per PR/issue at
``{root}/{forge_type}~{account}~{owner}~{repo}/{kind}-{number}.json``,
the durable record of pipeline state and review content. The operator
may edit or delete a file at state REVIEWED or later; earlier states
are owned by the running pipeline and may be overwritten at any time.

What belongs here: the schema and the load/save/read-modify-write of
these files. What does NOT belong: forge-fetched data (gcli_cache) and
review or queue logic (fairy, issue_fairy, pr_review_wrapper).
"""
from __future__ import annotations

import fcntl
import logging
import re
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Callable, Literal

# common's recursive JsonValue TypeAlias overflows pydantic's type
# resolution; pydantic's own JsonValue is the schema-capable equivalent.
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from common import atomic_write_text

__all__ = [
    "SCHEMA_VERSION",
    "WorkState",
    "LabelChange",
    "ReviewResult",
    "WorkItem",
    "item_path",
    "load_item",
    "save_item",
    "update_item",
    "logger",
]

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


class WorkState(IntEnum):
    QUEUED = 1      # passed the gates, waiting for / heading into the wrapper
    TRIAGE = 2      # wrapper: triage stage running
    REVIEW = 3      # wrapper: main reviewer pass running
    COMBINE = 4     # wrapper: combiner running, drafts recorded
    REVIEWED = 5    # final verdict present; awaiting operator / auto-post
    POSTED = 6
    SKIPPED = 7     # operator answered skip
    CANCELLED = 8   # operator threw it out
    ERROR = 9       # LLM/pipeline failure; error text recorded, never reused


class LabelChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    op: str  # "add" | "remove"
    reason: str = ""
    post: bool = False


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classification: str
    message: str
    label_changes: list[LabelChange] = []
    model: str = ""


class WorkItem(BaseModel):
    # ``extra="forbid"``: a typo in a hand-edited file must not be dropped.
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    kind: Literal["pr", "issue"]
    forge_type: str
    account: str = ""
    owner: str
    repo: str
    number: int
    state: WorkState
    created_at: str
    state_changed_at: str
    title: str = ""
    html_url: str = ""
    # Guard captured at review time: auto-posting is allowed only while
    # the live PR/issue still matches (nobody acted since the review).
    expected_updated_at: str | None = None
    expected_head_ref: str | None = None
    last_activity_iso: str | None = None
    consecutive_skip_count: int = 0
    triage: dict[str, JsonValue] | None = None
    drafts: list[ReviewResult] = []
    review: ReviewResult | None = None
    error: str | None = None

    def set_state(self, state: WorkState, now: datetime) -> None:
        self.state = state
        self.state_changed_at = now.isoformat()


def _segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value) or "default"


def item_path(
    root: Path,
    *,
    forge_type: str,
    account: str,
    owner: str,
    repo: str,
    kind: str,  # "pr" | "issue"
    number: int,
) -> Path:
    parts = [_segment(s) for s in (forge_type, account, owner, repo)]
    return root / "~".join(parts) / f"{kind}-{number}.json"


def load_item(path: Path) -> WorkItem | None:
    """None for a missing file (silent) or an unreadable/invalid one
    (logged as an error: the file may be a hand-edit gone wrong and must
    not be silently discarded)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("workset: no file %s", path)
        return None
    try:
        item = WorkItem.model_validate_json(text)
    except (ValidationError, ValueError) as exc:
        logger.error("workset: invalid item file %s: %s", path, exc)
        return None
    if item.schema_version != SCHEMA_VERSION:
        logger.error(
            "workset: %s has schema_version %d, expected %d; ignoring",
            path, item.schema_version, SCHEMA_VERSION,
        )
        return None
    return item


def save_item(path: Path, item: WorkItem) -> None:
    atomic_write_text(path, item.model_dump_json(indent=2) + "\n")
    logger.debug("workset: wrote %s state=%s", path, item.state.name)


def update_item(path: Path, mutate: Callable[[WorkItem], None]) -> WorkItem | None:
    """Read-modify-write ``path`` under an exclusive flock so concurrent
    updaters (pipeline transition vs operator edit via the TUI) cannot
    lose each other's changes. Returns the saved item, or None when the
    file is missing/invalid (the mutation is then not applied)."""
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        item = load_item(path)
        if item is None:
            return None
        mutate(item)
        save_item(path, item)
        return item
