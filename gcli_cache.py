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

Unified pickle cache for forgejo / gitea / github PR + issue data.
Used by fairy.py and forgejo_export.py.

Each entry is keyed on the server's ``updated_at``. Forges bump it on
every additive change (push, comment, review, label). Edits and
deletes of existing items do NOT bump ``updated_at`` on any supported
forge, so the three edit-prone fields (``issue_comments``,
``reviews``, ``review_comments``) additionally age out via
``max_age``. The stable-fetch loop in :func:`get` brackets the field
fetches between two ``updated_at`` reads and retries on advance, so
what we store is internally consistent with the stamped timestamp.
"""
from __future__ import annotations

import argparse
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import NamedTuple

from common import atomic_write_pickle, iso_to_dt
import forge_gcli

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4
MAX_REFETCH_ATTEMPTS = 3
EDIT_PRONE = frozenset({"issue_comments", "reviews", "review_comments"})


def _list_pr_files_or_empty_on_500(args, owner, repo, n):
    """Workaround for a forge API bug.

    ``GET /repos/.../pulls/{n}/files`` returns HTTP 500 (HTML error
    page, no JSON body) for PRs whose source fork was deleted -- the
    PR object then has ``head.repo=None`` and ``head.ref`` is the
    synthetic ``refs/pull/{n}/head``. The web UI still renders the
    file list for the same PRs, so the forge can compute the diff;
    only the API handler is broken. First observed on forgejo 15.0.2
    (~gitea 1.22.0); expected to persist across releases until
    upstream patches it. We lose the per-file metadata for those PRs
    (``forgejo_export`` then renders "_No file list found._") but
    keep the rest of the entry -- otherwise ``get``'s atomic-or-raise
    contract would discard timeline/commits/reviews too and re-fetch
    them every run.
    """
    try:
        return forge_gcli.list_pr_files(args, owner, repo, n)
    except RuntimeError as exc:
        if "code 500" not in str(exc):
            raise
        logger.warning(
            "gcli_cache: %s/%s pulls #%d /files returned 500; treating as "
            "empty (forge bug; see _list_pr_files_or_empty_on_500)",
            owner, repo, n,
        )
        return ()


# (kind, field_name) -> forge_gcli list function. ``kind`` is the URL
# segment, "pulls" or "issues". Membership is the valid-field check.
# ``review_comments`` is handled specially in ``_fetch_field`` because
# it consumes the freshly-fetched ``reviews`` list. ``list_issue_comments``
# is bound to its kind here -- GitLab needs distinct endpoints for
# PR vs Issue comments (forgejo/gitea/github share one), and even on
# the shared backends the kind shows up in the debug log line.
FETCH: dict[tuple[str, str], Callable] = {
    ("pulls",  "timeline"):        forge_gcli.list_issue_timeline,
    ("pulls",  "issue_comments"):  partial(forge_gcli.list_issue_comments, kind=forge_gcli.KIND_PR),
    ("pulls",  "reviews"):         forge_gcli.list_pr_reviews,
    ("pulls",  "review_comments"): forge_gcli.list_pr_review_comments,
    ("pulls",  "commits"):         forge_gcli.list_pr_commits,
    ("pulls",  "files"):           _list_pr_files_or_empty_on_500,
    ("issues", "timeline"):        forge_gcli.list_issue_timeline,
    ("issues", "issue_comments"):  partial(forge_gcli.list_issue_comments, kind=forge_gcli.KIND_ISSUE),
}
_FETCH_ORDER = ("timeline", "issue_comments", "reviews", "review_comments",
                "commits", "files")


class EntryKey(NamedTuple):
    """Identifies one cached PR or issue.

    ``forge_type`` and ``account`` are part of the key because gcli's
    ``-t``/``-a`` decide which forge instance an ``owner/repo`` pair
    resolves on -- two forges can serve the same name, and their
    payloads must not share a cache slot. ``workset.repo_dir`` keys its
    directories on the same pair.
    """
    forge_type: str
    account: str
    kind: str         # "pulls" or "issues"
    owner: str
    repo: str
    number: int


def entry_key(args: argparse.Namespace, kind: str, owner: str, repo: str,
              number: int) -> EntryKey:
    return EntryKey((args.forge_type or "").lower(), args.gcli_account or "",
                    kind, owner, repo, number)


@dataclass
class Entry:
    updated_at: datetime
    fetched_at: datetime
    comments_fetched_at: datetime | None      # TTL anchor for EDIT_PRONE
    fields: dict[str, tuple[dict, ...]] = field(default_factory=dict)


@dataclass
class Cache:
    version: int = SCHEMA_VERSION
    entries: dict[EntryKey, Entry] = field(default_factory=dict)


def load_cache(path: Path) -> Cache:
    try:
        obj = pickle.loads(path.read_bytes())
    except FileNotFoundError:
        return Cache()
    except Exception as exc:
        logger.warning("gcli_cache load %s: %s; empty cache", path, exc)
        return Cache()
    if isinstance(obj, Cache) and obj.version == SCHEMA_VERSION:
        logger.debug("gcli_cache load %s: %d entries", path, len(obj.entries))
        return obj
    logger.info("gcli_cache load %s: wrong shape/version; empty cache", path)
    return Cache()


def save_cache(path: Path, cache: Cache) -> None:
    atomic_write_pickle(path, cache)
    logger.debug("gcli_cache save %s: %d entries", path, len(cache.entries))


def _fetch_updated_at(args, kind: str, owner: str, repo: str, n: int) -> datetime:
    path = forge_gcli.build_repo_path(owner, repo, f"/{kind}/{n}")
    data = forge_gcli.gcli_api(args, path)
    if isinstance(data, dict):
        parsed = iso_to_dt(data.get("updated_at"))
        if parsed is not None:
            return parsed
    raise RuntimeError(f"gcli_cache: bad updated_at for {kind} #{n} ({path})")


def _fetch_field(args, kind: str, owner: str, repo: str, n: int,
                 name: str, reviews: list[dict] | None) -> tuple[dict, ...]:
    fn = FETCH[(kind, name)]
    items = (fn(args, owner, repo, n, reviews or [])
             if name == "review_comments" else fn(args, owner, repo, n))
    return tuple(items)


def get(
    cache: Cache,
    args: argparse.Namespace,
    kind: str,
    owner: str,
    repo: str,
    n: int,
    live_updated_at: datetime,
    *fields: str,
    max_age: timedelta,
    now: datetime | None = None,
    refetch: Callable = _fetch_updated_at,
) -> dict[str, tuple[dict, ...]]:
    """Return ``fields`` for one PR or issue.

    Cache hits (matching ``updated_at`` and, for edit-prone fields,
    within ``max_age``) are served directly. Misses go through the
    stable-fetch loop; if any edit-prone field is missed the rest of
    the trio is fetched too so they share a stamp. When ``updated_at``
    advanced, non-refetched siblings are dropped (next request
    refetches them).
    """
    bad = [f for f in fields if (kind, f) not in FETCH]
    if bad:
        raise ValueError(f"gcli_cache: unknown {kind} fields {bad}")
    if now is None:
        now = datetime.now(timezone.utc)
    key = entry_key(args, kind, owner, repo, n)
    old = cache.entries.get(key)

    def fresh(name: str) -> bool:
        if old is None or old.updated_at != live_updated_at or name not in old.fields:
            return False
        if name in EDIT_PRONE:
            return (old.comments_fetched_at is not None
                    and now - old.comments_fetched_at < max_age)
        return True

    missing = {f for f in fields if not fresh(f)}
    if not missing:
        logger.debug("gcli_cache hit %s %s/%s #%d: %s",
                     kind, owner, repo, n, list(fields))
        return {f: old.fields[f] for f in fields}

    valid_here = {f for (k, f) in FETCH if k == kind}
    to_fetch = missing | (EDIT_PRONE & valid_here if missing & EDIT_PRONE else set())
    ordered = [f for f in _FETCH_ORDER if f in to_fetch]
    logger.info("gcli_cache miss %s %s/%s #%d: fetching %s",
                kind, owner, repo, n, ordered)

    t0 = live_updated_at
    got: dict[str, tuple[dict, ...]] = {}
    for attempt in range(1, MAX_REFETCH_ATTEMPTS + 1):
        got = {}
        for name in ordered:
            reviews = list(got.get("reviews", ())) if name == "review_comments" else None
            got[name] = _fetch_field(args, kind, owner, repo, n, name, reviews)
        t1 = refetch(args, kind, owner, repo, n)
        if t1 == t0:
            break
        logger.info("gcli_cache: %s #%d updated_at %s -> %s; retry %d/%d",
                    kind, n, t0.isoformat(), t1.isoformat(),
                    attempt, MAX_REFETCH_ATTEMPTS)
        t0 = t1
    else:
        logger.warning("gcli_cache: %s #%d updated_at still moving after %d attempts",
                       kind, n, MAX_REFETCH_ATTEMPTS)

    keep_siblings = old is not None and old.updated_at == t0
    merged = dict(got)
    if keep_siblings:
        for name, value in old.fields.items():
            merged.setdefault(name, value)
    stamp = old.comments_fetched_at if (keep_siblings and old) else None
    if to_fetch & EDIT_PRONE:
        stamp = now
    cache.entries[key] = Entry(t0, now, stamp, merged)
    return {f: merged[f] for f in fields}
