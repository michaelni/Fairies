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

The filedb tickets' shared file plumbing.

The pipeline state itself lives in filedb (state = directory); what
remains here is the path layout shared with the gcli caches and the
sidecar-locked dict-level read-modify-write the wrapper (stage notes,
cancel flag) and the TUI use on claimed tickets.

What belongs here: path layout and update_json. What does NOT belong:
forge-fetched data (gcli_cache) and review or queue logic (fairy,
issue_fairy, pr_review_wrapper).
"""
from __future__ import annotations

import fcntl
import json
import logging
import re
from pathlib import Path
from typing import Callable

__all__ = [
    "repo_dir",
    "update_json",
    "logger",
]

logger = logging.getLogger(__name__)


def _segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value) or "default"


def repo_dir(root: Path, *, forge_type: str, account: str, owner: str, repo: str) -> Path:
    parts = [_segment(s) for s in (forge_type, account, owner, repo)]
    return root / "~".join(parts)


def update_json(path: Path, mutate: Callable[[dict], None]) -> dict | None:
    """Read-modify-write a ticket dict under an exclusive flock so
    concurrent updaters (the wrapper's stage notes vs the TUI's cancel)
    cannot lose each other's changes. Returns the saved dict, or None
    when the file is missing/unparseable (the mutation is then not
    applied)."""
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        mutate(data)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        return data
