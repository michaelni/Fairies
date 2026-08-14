#!/usr/bin/env python3
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

Bisect a poisoned vector store to name the file(s) that break ``files.list``.

A large store can acquire a corrupted index page where ``files.list`` 500s
forever past some cursor (see ``openai_vector_store_probe_poison.py``). The API
cannot name the offending entry. This tool reproduces the poison in throwaway
stores built from *existing* file ids (no re-upload) and binary-searches:

  1. Build a temp store from the candidate ids, confirm it 500s on a full list.
  2. Split the ids in half; rebuild and test each half.
  3. Recurse into every poisoned half until single ids remain.

If a set poisons but neither half does, the failure is size/interaction
dependent rather than caused by one entry; that set is reported as-is.

Throwaway stores get a 1-day expiry as an orphan safety net and are deleted
after each test unless ``--keep`` is set. File objects are shared and left
alone. Every API call is logged because any of them may 500.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from common import add_color_arg, setup_logging
from openai_common import load_api_key
from openai_vector_store_probe_poison import delete_entry, describe_error, parse_store_uploads

logger = logging.getLogger("vs_bisect")

Item = tuple[str, str]  # (path, file_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vector-store-id", required=True, help="Source store whose upload order is mined from the log.")
    p.add_argument(
        "--log",
        default=str(Path.home() / "forgejo_fairy" / "fairy_git_vector_sync.log"),
        help="Sync log to mine for the candidate file ids (in upload order).",
    )
    p.add_argument("--start", type=int, default=0, help="Slice the upload order from this index (default 0).")
    p.add_argument("--end", type=int, help="Slice the upload order up to this index (default end).")
    p.add_argument("--batch-size", type=int, default=100, help="File ids per attach batch (default 100).")
    p.add_argument("--poll-timeout", type=float, default=180.0, help="Seconds to wait for a batch to finish indexing.")
    p.add_argument("--list-attempts", type=int, default=1, help="Tries per list page before declaring poison; >1 guards against transient 500s (default 1).")
    p.add_argument("--keep", action="store_true", help="Do not delete the throwaway stores.")
    p.add_argument("--yes", action="store_true", help="Delete the identified poison files without prompting.")
    p.add_argument("--no-delete", action="store_true", help="Never delete identified poison files.")
    add_color_arg(p)
    return p.parse_args()


def attach_files(client: OpenAI, vs: str, file_ids: list[str], *, batch_size: int, poll_timeout: float) -> None:
    for start in range(0, len(file_ids), batch_size):
        chunk = file_ids[start:start + batch_size]
        batch = client.vector_stores.file_batches.create(vector_store_id=vs, file_ids=chunk)
        batch_id = batch.id  # capture once; retrieve responses echo a wrong id field
        logger.debug("batch created id=%s status=%s", batch_id, batch.status)
        deadline = time.monotonic() + poll_timeout
        while batch.status == "in_progress" and time.monotonic() < deadline:
            time.sleep(2)
            batch = client.vector_stores.file_batches.retrieve(batch_id, vector_store_id=vs)
        c = batch.file_counts
        logger.info(
            "attached chunk [%d:%d) status=%s completed=%d failed=%d in_progress=%d",
            start, start + len(chunk), batch.status, c.completed, c.failed, c.in_progress,
        )


def store_lists_clean(client: OpenAI, vs: str, *, attempts: int) -> bool:
    """Paginate the whole store. False == a page 500s persistently (poisoned)."""
    after: str | None = None
    seen = 0
    while True:
        err: dict | None = None
        for i in range(attempts):
            try:
                kwargs: dict[str, object] = {"vector_store_id": vs, "limit": 100, "order": "asc"}
                if after is not None:
                    kwargs["after"] = after
                page = client.vector_stores.files.list(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 -- boundary: classify any SDK error
                err = describe_error(exc)
                logger.info("list store=%s after=%s attempt %d/%d failed: %s", vs, after, i + 1, attempts, err)
                time.sleep(2 * (i + 1))
        else:
            logger.warning("POISONED store=%s listed=%d then 500 at after=%s %s", vs, seen, after, err)
            return False
        seen += len(page.data)
        logger.info("list store=%s page=%d total=%d has_more=%s", vs, len(page.data), seen, page.has_more)
        if not page.has_more or not page.last_id:
            return True
        after = page.last_id


def build_and_test(client: OpenAI, items: list[Item], *, args: argparse.Namespace) -> bool:
    """Build a throwaway store from ``items`` and return True if it lists clean."""
    store = client.vector_stores.create(
        name=f"poison-bisect-{len(items)}",
        expires_after={"anchor": "last_active_at", "days": 1},
    )
    logger.info("built store=%s for n=%d", store.id, len(items))
    attach_files(client, store.id, [fid for _, fid in items], batch_size=args.batch_size, poll_timeout=args.poll_timeout)
    clean = store_lists_clean(client, store.id, attempts=args.list_attempts)
    logger.info("store=%s n=%d -> %s", store.id, len(items), "clean" if clean else "POISONED")
    if not args.keep:
        client.vector_stores.delete(store.id)
        logger.info("deleted store=%s", store.id)
    return clean


def bisect(client: OpenAI, items: list[Item], *, args: argparse.Namespace) -> list[Item]:
    """Isolate poisoning entries in ``items`` (precondition: ``items`` poisons)."""
    if len(items) == 1:
        logger.info("isolated poison: path=%s file_id=%s", items[0][0], items[0][1])
        return items
    mid = len(items) // 2
    logger.info("=== bisecting n=%d into %d + %d ===", len(items), mid, len(items) - mid)
    poisoned_halves = [h for h in (items[:mid], items[mid:]) if not build_and_test(client, h, args=args)]
    logger.info("n=%d -> %d poisoned half(s)", len(items), len(poisoned_halves))
    if not poisoned_halves:
        logger.warning("EMERGENT: %d entries poison jointly but neither half does", len(items))
        return items
    found: list[Item] = []
    for half in poisoned_halves:
        found += bisect(client, half, args=args)
    return found


def main() -> int:
    args = parse_args()
    setup_logging(logger, verbose=True, color=args.color)

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("missing OPENAI_API_KEY in environment or .env")
    client = OpenAI(api_key=api_key)

    items = parse_store_uploads(Path(args.log), args.vector_store_id)[args.start:args.end]
    logger.info("bisecting %d candidate files [%d:%s)", len(items), args.start, args.end)
    if not items:
        raise SystemExit("no candidate files for the given slice")

    if build_and_test(client, items, args=args):
        logger.warning("candidate set does NOT reproduce the poison; nothing to bisect")
        return 1

    poison = bisect(client, items, args=args)
    logger.info("=== identified %d poisoning entries ===", len(poison))
    for path, fid in poison:
        logger.info("POISON path=%s file_id=%s", path, fid)

    report = Path(args.log).with_name("vector_store_bisect_report.json")
    report.write_text(json.dumps([{"path": p, "file_id": f} for p, f in poison], indent=2), encoding="utf-8")
    logger.info("wrote report %s", report)

    if poison and not args.no_delete:
        if args.yes or input(f"delete {len(poison)} poison file(s) from {args.vector_store_id}? [y/N] ").strip().lower() == "y":
            for _, fid in poison:
                delete_entry(client, args.vector_store_id, fid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
