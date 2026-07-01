#!/usr/bin/env python3
"""Audit a poisoned vector store and offer to remove unreadable entries.

Large vector stores (~5000+ files) sometimes acquire a corrupted index page:
``files.list`` paginated past a certain ``after=`` cursor returns a server-side
500 forever, which crashes the incremental sync. The store still works for
retrieval; only enumeration is broken. The bad entry cannot be named by the API
directly -- the ``files.list`` call that would return it is the one that 500s --
so we recover ids from the *upload order* in the sync log (``files.list`` is
``created_at`` ascending == upload order).

Steps, all logged in full because any call may 500:
  1. Print the store's aggregate ``file_counts`` (interesting on their own).
  2. Paginate the whole store. On a page error, step one id at a time (limit=1)
     using the next id from the log to resume; an id whose step-1 read 500s is
     recorded as unreadable and skipped. Repeat until the store is exhausted.
  3. Audit every batch (ids from the log; there is no batch-list API), logging
     failures, cancellations, in-progress, and odd sizes.
  4. List the unreadable file ids with their paths and offer to delete them.

A JSON report is written next to the log.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from common import add_color_arg, setup_logging
from openai_common import load_api_key

logger = logging.getLogger("vs_probe")

# ``files.create ok <path> <file-id>`` short form. The verbose form is
# ``files.create ok what=... path=... file_id=... dt=...``; it never ends in a
# bare ``file-<id>`` token so this regex skips it.
UPLOAD_RE = re.compile(r"files\.create ok (?P<path>\S+) (?P<fid>file-\w+)\s*$")
BATCH_RE = re.compile(r"file_batches\.create ok vector_store_id=(?P<vs>vs_\w+) batch_id=(?P<bid>vsfb_\w+)")

PROD_PAGE_SIZE = 100  # matches list_vector_store_files in openai_vector_store.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vector-store-id", required=True, help="The poisoned vector store id.")
    p.add_argument(
        "--log",
        default=str(Path.home() / "forgejo_fairy" / "fairy_git_vector_sync.log"),
        help="Sync log to mine for upload order and batch ids.",
    )
    p.add_argument("--yes", action="store_true", help="Delete unreadable entries without prompting.")
    p.add_argument("--no-delete", action="store_true", help="Never delete; audit only.")
    p.add_argument("--purge-around", type=int, metavar="N", help="Delete upload-ordered ids in a window centred on index N (assumes list order ~ upload order).")
    p.add_argument("--purge-radius", type=int, default=100, help="Half-width of --purge-around window (default 100).")
    add_color_arg(p)
    return p.parse_args()


def parse_store_uploads(log_path: Path, vector_store_id: str) -> list[tuple[str, str]]:
    """Return ``(path, file_id)`` in upload order for *this* store only.

    The log interleaves uploads for every store it has ever built, so a plain
    scan would mix in thousands of unrelated ids. Uploads are emitted in a chunk
    and then attached by the next ``file_batches.create``, so each pending chunk
    belongs to the store named on the following batch line.
    """
    order: list[tuple[str, str]] = []
    pending: list[tuple[str, str]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        mu = UPLOAD_RE.search(line)
        if mu:
            pending.append((mu.group("path"), mu.group("fid")))
            continue
        mb = BATCH_RE.search(line)
        if mb:
            if mb.group("vs") == vector_store_id:
                order.extend(pending)
            pending.clear()
    logger.info("parsed %d uploaded file ids for %s", len(order), vector_store_id)
    return order


def parse_batch_ids(log_path: Path, vector_store_id: str) -> list[str]:
    ids: list[str] = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = BATCH_RE.search(line)
        if m and m.group("vs") == vector_store_id:
            ids.append(m.group("bid"))
    logger.info("parsed %d batch ids for %s", len(ids), vector_store_id)
    return ids


def describe_error(exc: Exception) -> dict[str, object]:
    info: dict[str, object] = {"type": type(exc).__name__, "message": str(exc)}
    for attr in ("status_code", "request_id"):
        val = getattr(exc, attr, None)
        if val is not None:
            info[attr] = val
    return info


def safe_list(client: OpenAI, vs: str, cursor: str | None, limit: int) -> tuple[object | None, dict | None]:
    """One raw ``files.list`` page. Returns ``(page, None)`` or ``(None, err)``.

    No retry wrapper: a persistent 500 is exactly the signal we probe for.
    """
    kwargs: dict[str, object] = {"vector_store_id": vs, "limit": limit, "order": "asc"}
    if cursor is not None:
        kwargs["after"] = cursor
    started = time.monotonic()
    try:
        page = client.vector_stores.files.list(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- boundary: classify any SDK error
        err = describe_error(exc)
        logger.debug("files.list after=%s limit=%d FAILED dt=%.1fs %s", cursor, limit, time.monotonic() - started, err)
        return None, err
    logger.debug("files.list after=%s limit=%d ok returned=%d has_more=%s", cursor, limit, len(page.data), page.has_more)
    return page, None


def print_aggregate_counts(client: OpenAI, vs: str) -> dict[str, object]:
    store = client.vector_stores.retrieve(vs)
    c = store.file_counts
    counts = {
        "status": store.status,
        "usage_bytes": store.usage_bytes,
        "total": c.total,
        "completed": c.completed,
        "in_progress": c.in_progress,
        "failed": c.failed,
        "cancelled": c.cancelled,
    }
    logger.info("aggregate counts %s", counts)
    return counts


def paginate_with_recovery(client: OpenAI, vs: str, order: list[tuple[str, str]], total: int) -> tuple[set[str], list[dict]]:
    """Walk the whole store, resuming past unreadable pages.

    Normal pages use ``limit=100``. When a page 500s, switch to single-id steps
    (``limit=1``) keyed off the log's upload order: each healthy step is the next
    expected id, and a step whose read 500s names the unreadable id (the API
    cannot return it, but the log can). Recording it lets us resume past it.
    """
    id_to_path = {fid: path for path, fid in order}
    ids_order = [fid for _, fid in order]
    pos = {fid: i for i, fid in enumerate(ids_order)}
    known = set(ids_order)

    returned: set[str] = set()      # ids the API actually listed (reachable)
    failures: list[dict] = []       # ids whose step-1 read 500s (unreadable)
    failing: set[str] = set()
    accounted: set[str] = set()     # returned | failing; used to advance past handled ids

    def to_test() -> int:
        return len(known - accounted)

    def next_unaccounted(cursor: str | None) -> str | None:
        start = 0 if cursor is None else pos.get(cursor, -1) + 1
        for j in range(start, len(ids_order)):
            if ids_order[j] not in accounted:
                return ids_order[j]
        return None

    def mark_returned(fid: str) -> None:
        returned.add(fid)
        accounted.add(fid)

    cursor: str | None = None
    while True:
        page, _ = safe_list(client, vs, cursor, PROD_PAGE_SIZE)
        if page is not None:
            for f in page.data:
                mark_returned(f.id)
            logger.info("paginated returned=%d failing=%d to_test=%d (total=%s)", len(returned), len(failing), to_test(), total)
            if not page.has_more:
                break
            cursor = page.data[-1].id
            continue

        # Bulk page failed: step across the bad region one id at a time.
        crossed = False
        while True:
            one, oerr = safe_list(client, vs, cursor, 1)
            if one is not None:
                if not one.data:
                    return returned, failures
                got = one.data[0].id
                mark_returned(got)
                cursor = got
                if not one.has_more:
                    return returned, failures
                if crossed:
                    break  # past the poison; resume bulk pagination
                continue
            # The next-by-upload-order id names the unreadable entry; this is
            # approximate because list order is created_at+id, not upload order.
            bad = next_unaccounted(cursor)
            if bad is None:
                return returned, failures
            failures.append({"file_id": bad, "path": id_to_path.get(bad), **oerr})
            failing.add(bad)
            accounted.add(bad)
            logger.warning("unreadable after %s: %s (%s) status=%s", cursor, bad, id_to_path.get(bad), oerr.get("status_code"))
            cursor = bad
            crossed = True

    return returned, failures


def audit_batches(client: OpenAI, vs: str, batch_ids: list[str]) -> list[dict]:
    results: list[dict] = []
    for bid in batch_ids:
        try:
            b = client.vector_stores.file_batches.retrieve(bid, vector_store_id=vs)
        except Exception as exc:  # noqa: BLE001 -- boundary: classify any SDK error
            logger.warning("batch %s retrieve FAILED %s", bid, describe_error(exc))
            results.append({"batch_id": bid, "error": describe_error(exc)})
            continue
        c = b.file_counts
        row = {
            "batch_id": bid,
            "status": b.status,
            "total": c.total,
            "completed": c.completed,
            "failed": c.failed,
            "cancelled": c.cancelled,
            "in_progress": c.in_progress,
        }
        odd = c.failed or c.cancelled or c.in_progress or b.status != "completed" or c.completed != c.total
        (logger.warning if odd else logger.info)("batch %s", row)
        results.append(row)
    return results


def delete_entry(client: OpenAI, vs: str, file_id: str) -> bool:
    """Detach from the store AND delete the underlying file object.

    ``vector_stores.files.delete`` only detaches and leaves a stuck ghost; the
    file is only truly gone once ``files.delete`` removes the underlying object
    (https://community.openai.com/t/deleting-vector-store-files-does-not-delete-them/1091902).
    Returns True only if both calls succeed.
    """
    ok = True
    try:
        res = client.vector_stores.files.delete(file_id, vector_store_id=vs)
        logger.info("detach ok file_id=%s deleted=%s", file_id, res.deleted)
    except Exception as exc:  # noqa: BLE001 -- boundary: classify any SDK error
        logger.error("detach FAILED file_id=%s %s", file_id, describe_error(exc))
        ok = False
    try:
        res = client.files.delete(file_id)
        logger.info("files.delete ok file_id=%s deleted=%s", file_id, res.deleted)
    except Exception as exc:  # noqa: BLE001 -- boundary: classify any SDK error
        logger.error("files.delete FAILED file_id=%s %s", file_id, describe_error(exc))
        ok = False
    return ok


def purge_window(client: OpenAI, vs: str, order: list[tuple[str, str]], center: int, radius: int) -> None:
    """Delete a ``+-radius`` window of upload-ordered ids around ``center``.

    The exact poison id cannot be named (``files.list`` 500s there), so when list
    order tracks upload order this clears the corrupt region by index instead.
    """
    ids = [fid for _, fid in order]
    lo = max(0, center - radius)
    hi = min(len(ids), center + radius + 1)
    logger.info("purging upload-order window [%d:%d) of %d ids", lo, hi, len(ids))
    for idx in range(lo, hi):
        logger.info("--- window idx %d ---", idx)
        delete_entry(client, vs, ids[idx])
    logger.warning("purged window [%d:%d)", lo, hi)


def main() -> int:
    args = parse_args()
    setup_logging(logger, verbose=True, color=args.color)
    log_path = Path(args.log)

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("missing OPENAI_API_KEY in environment or .env")
    client = OpenAI(api_key=api_key)

    if args.purge_around is not None:
        order = parse_store_uploads(log_path, args.vector_store_id)
        logger.info("=== purge window around index %d (+-%d) ===", args.purge_around, args.purge_radius)
        purge_window(client, args.vector_store_id, order, args.purge_around, args.purge_radius)
        return 0

    order = parse_store_uploads(log_path, args.vector_store_id)
    batch_ids = parse_batch_ids(log_path, args.vector_store_id)

    report: dict[str, object] = {"vector_store_id": args.vector_store_id}
    aggregate = print_aggregate_counts(client, args.vector_store_id)
    report["aggregate"] = aggregate

    logger.info("=== paginating store with recovery ===")
    returned, failures = paginate_with_recovery(client, args.vector_store_id, order, aggregate["total"])
    known = {fid for _, fid in order}
    failing_ids = {f["file_id"] for f in failures}
    missing = known - returned - failing_ids
    summary = {
        "returned": len(returned),
        "missing": len(missing),
        "failing": len(failing_ids),
        "to_be_tested": len(known - returned - failing_ids - missing),
    }
    logger.info("pagination done: %s (store total=%s, log known=%d)", summary, aggregate["total"], len(known))
    report["summary"] = summary
    report["unreadable"] = failures
    report["missing"] = sorted(missing)

    logger.info("=== auditing %d batches ===", len(batch_ids))
    report["batches"] = audit_batches(client, args.vector_store_id, batch_ids)

    report_path = log_path.with_name("vector_store_poison_probe_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("wrote report %s", report_path)

    if not failures:
        logger.info("no unreadable entries found")
        return 0

    logger.warning("unreadable entries (%d):", len(failures))
    for f in failures:
        logger.warning("  %s  %s  status=%s", f["file_id"], f.get("path"), f.get("status_code"))

    if args.no_delete:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            logger.warning("stdin not a tty; pass --yes to delete or --no-delete to silence")
            return 0
        if input(f"delete these {len(failures)} entries? [y/N] ").strip().lower() != "y":
            logger.info("not deleting")
            return 0

    for f in failures:
        delete_entry(client, args.vector_store_id, f["file_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
