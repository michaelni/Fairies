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

The per-author settle time: how long a push or reply by an author must
settle before fairy re-reviews an item she has already engaged on.
Authors whose pull requests draw many fairy reviews per landed pull
request wait longer than the default.

What belongs here: the reviews-per-landed ratio over the items store,
the patch-id index of the forge's branches that tells a hand-applied
pull request from an abandoned one, and the ratio-to-hours steps.
What does NOT belong: the gate that applies the hours (fairy,
issue_fairy) and the pass that triggers the computation (agent).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import filedb
import git_util

__all__ = ["REVIEWS_PER_LANDED_STEPS", "landed_patch_ids", "settle_hours"]

logger = logging.getLogger(__name__)

REVIEWS_PER_LANDED_STEPS = ((12, 48.0), (10, 24.0), (8, 12.0))

LANDED_DOC = "landed-patch-ids.json"


def landed_patch_ids(db: filedb.Db, repo_root: Path, since: datetime) -> set[str]:
    """The patch-ids of the forge's branches, pull request refs excluded.

    db:        the db whose root caches the sha-to-patch-id map
    repo_root: a clone with the forge remote (see git_util.FORGE_REMOTES)
    since:     only non-merge commits from this time on
    returns:   the patch-ids
    """
    remote = git_util.git_forge_remote(repo_root)
    shas = git_util.git_rev_list(
        repo_root, f"--since={since.isoformat()}", "--no-merges",
        f"--exclude={remote}/pr/*", f"--remotes={remote}")
    known = db.read(db.root / LANDED_DOC) or {}
    ids = {sha: known[sha] for sha in shas if sha in known}
    new = [sha for sha in shas if sha not in known]
    if new:
        ids.update(git_util.git_patch_ids(repo_root, new))
        db.write(db.root / LANDED_DOC, ids)
    logger.debug("landed patch-ids: %d commits since %s, %d newly hashed",
                 len(shas), since.date(), len(new))
    return set(ids.values())


def settle_hours(db: filedb.Db, reviewer: str, landed: set[str]) -> dict[str, float]:
    """Hours a push or reply by an author must settle before a re-review.

    db:       the db whose items store holds the pull requests
    reviewer: fairy's forge login; her reviews and comments are counted
    landed:   patch-ids on the forge's branches; a pull request has
              landed when the forge merged it or all its patch-ids are in here
    returns:  {author: hours} for every author whose reviews per landed
              pull request plus one reach a REVIEWS_PER_LANDED_STEPS ratio
    """
    reviews: Counter[str] = Counter()
    landed_prs: Counter[str] = Counter()
    for kind, number in db.list_state(filedb.ITEM_STATE):
        item = db.get(filedb.ITEM_STATE, kind, number) if kind == "pr" else None
        if not item or not item.get("author"):
            continue
        author = item["author"]
        reviews[author] += sum(
            entry.get("author") == reviewer and entry.get("kind") in ("review", "comment")
            for entry in item.get("discussion") or ())
        ids = item.get("patch_ids")
        landed_prs[author] += item.get("state") == "merged" or bool(ids) and all(
            pid in landed for pid in ids)
    hours: dict[str, float] = {}
    for author, count in reviews.items():
        ratio = count / (landed_prs[author] + 1)
        for step, step_hours in REVIEWS_PER_LANDED_STEPS:
            if ratio >= step:
                hours[author] = step_hours
                logger.info("settle hours: %s waits %gh (%d reviews, %d landed, ratio %.1f)",
                            author, step_hours, count, landed_prs[author], ratio)
                break
    return hours
