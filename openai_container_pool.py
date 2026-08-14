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

Filesystem-backed pool of interchangeable OpenAI container ids.

Every file inside a pool directory names a container_id believed to be
ready and idle. Claim = ``os.unlink`` (the first concurrent caller
wins); release = ``O_CREAT|O_EXCL`` put-back. That is the entire
concurrency story.

The pool helpers themselves do no liveness check; the caller
(``ensure_openai_container_repos_ready``) verifies each claimed
container with ``get_live_container_id`` and skips stale entries
before returning. If a container that passed the liveness check dies
between then and first use, the caller releases with ``healthy=False``
and the entry stays gone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


POOL_SCHEMA_VERSION = "v1"
DEFAULT_POOL_ROOT = Path.home() / ".cache" / "forgejo_fairy" / "container_pool"

_CONTAINER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,128}$")
_STATE_HASH_RE = re.compile(r"^[0-9a-f]{4,64}$")


def compute_container_state_hash(state: object) -> str:
    """16-hex-char sha256 prefix over a canonical JSON encoding of ``state``."""
    encoded = json.dumps(
        {"schema": POOL_SCHEMA_VERSION, "state": state},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def pool_dir_for(pool_root: Path, state_hash: str) -> Path:
    if not _STATE_HASH_RE.match(state_hash):
        raise ValueError(f"invalid state_hash: {state_hash!r}")
    return pool_root / state_hash


def try_claim_container(pool_dir: Path) -> str | None:
    """Unlink one pool entry and return its filename, or ``None`` if empty.

    Missing pool directory, missing entries (lost a race with another
    claimer), and non-matching filenames all fold into ``None``/skip.
    """
    try:
        names = os.listdir(pool_dir)
    except FileNotFoundError:
        return None
    for name in names:
        if not _CONTAINER_ID_RE.match(name):
            continue
        try:
            os.unlink(pool_dir / name)
        except FileNotFoundError:
            continue
        logger.debug("container pool claim ok pool_dir=%s container_id=%s", pool_dir, name)
        return name
    logger.debug("container pool claim empty pool_dir=%s", pool_dir)
    return None


def release_container(pool_dir: Path, container_id: str, *, healthy: bool) -> bool:
    """Put ``container_id`` back into the pool iff ``healthy``.

    Returns ``True`` iff a pool entry was created.
    """
    if not isinstance(container_id, str) or not _CONTAINER_ID_RE.match(container_id):
        logger.warning("container pool release skipped: malformed container_id=%r", container_id)
        return False
    if not healthy:
        logger.debug(
            "container pool release discard pool_dir=%s container_id=%s",
            pool_dir, container_id,
        )
        return False
    pool_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(pool_dir / container_id, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        logger.warning(
            "container pool release: entry already exists pool_dir=%s container_id=%s",
            pool_dir, container_id,
        )
        return False
    os.close(fd)
    logger.debug("container pool release ok pool_dir=%s container_id=%s", pool_dir, container_id)
    return True
