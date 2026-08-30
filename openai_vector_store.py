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

Vector store helpers for the PR review wrapper.

All logic for building, uploading to, and syncing an OpenAI vector store for
the PR review wrapper lives here. The wrapper imports the public entry points
(``ensure_vector_stores_synced_for_repos`` and
``get_live_cached_vector_stores_for_repos``) and delegates to them.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import TypeAlias

from openai import OpenAI
from openai.types.vector_stores.vector_store_file import LastError

from common import atomic_write_text
from git_util import (
    get_repo_head_sha,
    git_show_file_bytes,
    list_head_tree_entries,
)
from openai_common import (
    call_with_rate_limit_retry,
    log_progress,
    upload_local_file,
)


logger = logging.getLogger(__name__)


VectorStoreCache: TypeAlias = dict[str, str | None]
VectorStoreBlobEntry: TypeAlias = dict[str, object]
DEFAULT_VECTOR_STORE_EXPIRY_DAYS = 30
DEFAULT_VECTOR_STORE_FILE_BATCH_SIZE = 50 # 50 seems more stable than 100 but it may be a coincidence try 100 again in a few days
DEFAULT_VECTOR_STORE_FILE_BATCH_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_VECTOR_STORE_FILE_BATCH_POLL_TIMEOUT_SECONDS = 300.0
DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS = 3
DEFAULT_VECTOR_STORE_SYNC_MAX_RETRIES = 0
VECTOR_STORE_CACHE_FILENAME = ".openai_vector_store_cache.json"
# Extensions OpenAI file search accepts for retrieval
# (https://developers.openai.com/api/docs/guides/tools-file-search);
# vector_stores.file_batches.create rejects any other extension with a
# 400 ``unsupported_file``, so everything else must be wrapped as .txt
# or skipped.
SUPPORTED_VECTOR_STORE_EXTENSIONS = {
    "c",
    "cpp",
    "cs",
    "css",
    "doc",
    "docx",
    "go",
    "html",
    "java",
    "js",
    "json",
    "md",
    "pdf",
    "php",
    "pptx",
    "py",
    "rb",
    "sh",
    "tex",
    "ts",
    "txt",
}

TEXT_WRAPPED_VECTOR_STORE_EXTENSIONS = {
    "",
    "art",
    "asm",
    "awk",
    "bat",
    "brf",
    "cgi",
    "cl",
    "cls",
    "cnf",
    "csv",
    "cu",
    "cuh",
    "diff",
    "dot",
    "eml",
    "es",
    "example",
    "exr",
    "ffconcat",
    "ffmeta",
    "ffpreset",
    "gif",
    "glsl",
    "h",
    "hpp",
    "hs",
    "htm",
    "hwp",
    "hwpx",
    "ics",
    "ifb",
    "init",
    "jpeg",
    "jpg",
    "keynote",
    "ksh",
    "ltx",
    "m",
    "mail",
    "mak",
    "manifest",
    "markdown",
    "metal",
    "mht",
    "mhtml",
    "mjs",
    "nws",
    "odt",
    "pages",
    # "pam",
    "patch",
    "pkl",
    "pl",
    "pm",
    "png",
    "pot",
    "ppa",
    "pps",
    "ppt",
    "pwz",
    "rc",
    "rst",
    "rtf",
    "s",
    "scala",
    "shtml",
    "srt",
    "sty",
    "supp",
    "tar",
    "template",
    "texi",
    "text",
    "v",
    "vcf",
    "voc",
    "vtt",
    "webp",
    "wiz",
    "xla",
    "xlb",
    "xlc",
    "xlm",
    "xls",
    "xlsx",
    "xlt",
    "xlw",
    "xml",
    "xsd",
    "xwd",
    "y4m",
    "yaml",
    "yml",
    "zip",
}


def supported_vector_store_extensions(
    client: OpenAI,
    repo_root: Path,
    *,
    verbose: bool,
) -> set[str]:
    return SUPPORTED_VECTOR_STORE_EXTENSIONS


def get_vector_store_upload_suffix(relpath: str, supported_extensions: set[str]) -> str | None:
    suffix = Path(relpath).suffix
    normalized = suffix.lstrip(".").lower() if suffix else ""
    if normalized in supported_extensions:
        return normalized
    if normalized in TEXT_WRAPPED_VECTOR_STORE_EXTENSIONS:
        return "txt"
    return None


def build_vector_store_upload_bytes(
    relpath: str,
    blob_sha: str,
    data: bytes,
    upload_suffix: str,
) -> bytes:
    if upload_suffix != "txt":
        return data

    suffix = Path(relpath).suffix
    normalized = suffix.lstrip(".").lower() if suffix else ""
    header = (
        f"ORIGINAL_PATH: {relpath}\n"
        f"ORIGINAL_EXTENSION: {suffix or '-'}\n"
        f"BLOB_SHA: {blob_sha}\n"
    )

    if normalized == "png" or b"\x00" in data:
        body = base64.b64encode(data).decode("ascii")
        return (
            header
            + "CONTENT_FORMAT: base64\n\n"
            + body
            + "\n"
        ).encode("utf-8")

    text = data.decode("utf-8", errors="replace")
    return (
        header
        + "CONTENT_FORMAT: text\n\n"
        + text
    ).encode("utf-8")


def normalize_vector_store_cache(data: object) -> VectorStoreCache:
    if not isinstance(data, dict):
        return {"vector_store_id": None}

    vector_store_id = data.get("vector_store_id")
    if not isinstance(vector_store_id, str) or not vector_store_id:
        vector_store_id = None

    return {"vector_store_id": vector_store_id}


def load_vector_store_cache(repo_root: Path) -> VectorStoreCache:
    cache_path = repo_root / VECTOR_STORE_CACHE_FILENAME
    if not cache_path.exists():
        return {"vector_store_id": None}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("%s", exc)
        return {"vector_store_id": None}
    return normalize_vector_store_cache(data)


def save_vector_store_cache(repo_root: Path, cache: VectorStoreCache) -> None:
    cache_path = repo_root / VECTOR_STORE_CACHE_FILENAME
    atomic_write_text(
        cache_path,
        json.dumps(normalize_vector_store_cache(cache), indent=2, sort_keys=True),
    )


def build_vector_store_temp_upload_path(
    temp_dir: Path,
    relpath: str,
    blob_sha: str,
    upload_suffix: str,
) -> Path:
    stem = Path(relpath).stem or "file"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:80] or "file"
    return temp_dir / f"{blob_sha[:16]}_{safe_stem}.{upload_suffix}"


def list_vector_store_files(
    client: OpenAI,
    vector_store_id: str,
    *,
    verbose: bool,
) -> list[object]:
    items: list[object] = []
    after: str | None = None

    seen = set()
    while True:
        if verbose:
            logger.debug("openai vector_stores.files.list start vector_store_id=%s after=%s", vector_store_id, after or "-")

        def fetch_list() -> object:
            kwargs: dict[str, object] = {"vector_store_id": vector_store_id, "limit": 100, "order" : "asc"}
            if after is not None:
                kwargs["after"] = after
            return client.vector_stores.files.list(**kwargs)

        page = call_with_rate_limit_retry(
            fetch_list,
            what=f"vector store files list for {vector_store_id}",
            verbose=verbose,
        )

        page_items = [x for x in page.data if not (x.id in seen or seen.add(x.id))]
        items.extend(page_items)

        if verbose:
            logger.debug("openai vector_stores.files.list ok vector_store_id=%s count=%d items=%d", vector_store_id, len(page_items), len(items))

        if not page.has_more:
            break

        after = page.last_id
        if not after:
            logger.warning("vector store files list has_more set but no last_id pointer")
            break

    return items


def list_vector_store_blob_entries(
    client: OpenAI,
    vector_store_id: str,
    *,
    verbose: bool,
) -> list[VectorStoreBlobEntry]:
    entries: list[VectorStoreBlobEntry] = []

    for item in list_vector_store_files(client, vector_store_id, verbose=verbose):
        attrs = item.attributes or {}
        blob_sha = attrs.get("blob_sha", "")
        if not blob_sha and verbose:
            logger.debug("vector store file without blob_sha attribute %s", item.id)

        entries.append({
            "blob_sha": blob_sha,
            "vector_store_file_id": item.id,
            "path": attrs.get("path", ""),
            "status": item.status,
            "last_error": item.last_error,
        })

    return entries


def format_vector_store_last_error(last_error: LastError | None) -> str:
    if last_error is None:
        return "-"
    return f"{last_error.code}: {last_error.message}"


def log_failed_vector_store_blobs(
    entries: list[VectorStoreBlobEntry],
    *,
    prefix: str,
) -> None:
    for entry in entries:
        logger.warning(
            "%s: status=%s path=%s blob_sha=%s vector_store_file_id=%s error=%s",
            prefix,
            entry.get("status", "unknown"),
            entry.get("path", ""),
            entry.get("blob_sha", "") or "-",
            entry.get("vector_store_file_id", ""),
            format_vector_store_last_error(entry.get("last_error")),
        )


def delete_vector_store_file(
    client: OpenAI,
    vector_store_id: str,
    vector_store_file_id: str,
    *,
    verbose: bool,
) -> None:
    if verbose:
        logger.debug("openai vector_stores.files.delete start vector_store_id=%s file_id=%s", vector_store_id, vector_store_file_id)
    call_with_rate_limit_retry(
        lambda: client.vector_stores.files.delete(
            vector_store_id=vector_store_id,
            file_id=vector_store_file_id,
        ),
        what=f"vector store file delete {vector_store_file_id}",
        verbose=verbose,
    )
    if verbose:
        logger.debug("openai vector_stores.files.delete ok vector_store_id=%s file_id=%s", vector_store_id, vector_store_file_id)


def attach_batch_once(
    client: OpenAI,
    vector_store_id: str,
    attach_items: list[tuple[str, str, str]],
    *,
    verbose: bool,
) -> tuple[str, dict[str, int]]:
    files_payload = [
        {"file_id": file_id, "attributes": {"path": relpath, "blob_sha": blob_sha}}
        for file_id, relpath, blob_sha in attach_items
    ]

    if verbose:
        logger.debug("openai vector_stores.file_batches.create start vector_store_id=%s files=%d", vector_store_id, len(files_payload))

    batch = call_with_rate_limit_retry(
        lambda: client.vector_stores.file_batches.create(
            vector_store_id=vector_store_id,
            files=files_payload,
        ),
        what=f"vector store file batch create for {vector_store_id}",
        verbose=verbose,
    )
    batch_id = batch.id

    if verbose:
        logger.debug("openai vector_stores.file_batches.create ok vector_store_id=%s batch_id=%s files=%d", vector_store_id, batch_id, len(files_payload))

    started = time.monotonic()
    while True:
        batch = call_with_rate_limit_retry(
            lambda: client.vector_stores.file_batches.retrieve(
                batch_id=batch_id,
                vector_store_id=vector_store_id,
            ),
            what=f"vector store file batch poll for {vector_store_id}",
            verbose=verbose,
        )
        counts = batch.file_counts
        status = batch.status
        result = {
            "completed": counts.completed,
            "failed": counts.failed,
            "cancelled": counts.cancelled,
            "in_progress": counts.in_progress,
            "timed_out": 0,
        }

        # Done only at a terminal batch ``status``. A just-created batch
        # briefly reports ``status=in_progress`` with all ``file_counts``
        # still zero (counts not yet populated); gating on ``in_progress
        # <= 0`` mistook that for completion and returned with
        # ``completed=0``, never confirming the files processed.
        if status in ("completed", "failed", "cancelled"):
            if verbose:
                logger.debug("openai vector_stores.file_batches.poll ok vector_store_id=%s batch_id=%s status=%s completed=%d failed=%d cancelled=%d in_progress=%d dt=%.3fs", vector_store_id, batch_id, status, counts.completed, counts.failed, counts.cancelled, counts.in_progress, time.monotonic() - started)
            return batch_id, result

        if time.monotonic() - started >= DEFAULT_VECTOR_STORE_FILE_BATCH_POLL_TIMEOUT_SECONDS:
            logger.warning(
                "vector store batch poll timed out: vector_store_id=%s batch_id=%s status=%s completed=%d failed=%d cancelled=%d in_progress=%d dt=%.3fs",
                vector_store_id, batch_id, status, counts.completed, counts.failed, counts.cancelled, counts.in_progress, time.monotonic() - started,
            )
            result["timed_out"] = 1
            return batch_id, result

        time.sleep(DEFAULT_VECTOR_STORE_FILE_BATCH_POLL_INTERVAL_SECONDS)


def list_batch_completed_file_ids(
    client: OpenAI,
    vector_store_id: str,
    batch_id: str,
    *,
    verbose: bool,
) -> set[str]:
    completed_ids: set[str] = set()
    after: str | None = None
    while True:
        def fetch_completed() -> object:
            kwargs: dict[str, object] = {
                "batch_id": batch_id,
                "vector_store_id": vector_store_id,
                "filter": "completed",
                "limit": 100,
            }
            if after is not None:
                kwargs["after"] = after
            return client.vector_stores.file_batches.list_files(**kwargs)

        page = call_with_rate_limit_retry(
            fetch_completed,
            what=f"vector store batch completed-file list for {batch_id}",
            verbose=verbose,
        )
        completed_ids.update(item.id for item in page.data)
        if not page.has_more:
            break
        after = page.last_id
        if not after:
            break
    return completed_ids


def create_vector_store_batch_and_poll(
    client: OpenAI,
    vector_store_id: str,
    attach_items: list[tuple[str, str, str]],
    *,
    verbose: bool,
) -> dict[str, int]:
    if not attach_items:
        return {"completed": 0, "failed": 0, "cancelled": 0, "in_progress": 0, "timed_out": 0}

    total = len(attach_items)
    remaining = attach_items

    for attempt in range(1, DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS + 1):
        batch_id, counts = attach_batch_once(client, vector_store_id, remaining, verbose=verbose)
        if not counts["timed_out"] and not counts["failed"] and not counts["cancelled"]:
            return {"completed": total, "failed": 0, "cancelled": 0, "in_progress": 0, "timed_out": 0}

        # Some files did not complete -- failed, cancelled, or still
        # in_progress when the poll timed out. Reconcile against what
        # actually completed (one batch-scoped list), drop the incomplete
        # entries so they cannot linger as poison, and re-attach the rest. A
        # timeout from a transient API outage usually reconciles to "all
        # completed" here once the API recovers.
        completed_ids = list_batch_completed_file_ids(client, vector_store_id, batch_id, verbose=verbose)
        remaining = [item for item in remaining if item[0] not in completed_ids]
        for file_id, _relpath, _blob_sha in remaining:
            delete_vector_store_file(client, vector_store_id, file_id, verbose=verbose)

        if not remaining:
            return {"completed": total, "failed": 0, "cancelled": 0, "in_progress": 0, "timed_out": 0}
        if attempt == DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS:
            logger.warning("vector store batch giving up on %d incomplete file(s) after %d attempt(s): %s", len(remaining), attempt, ", ".join(item[1] for item in remaining))
            return {"completed": total - len(remaining), "failed": len(remaining), "cancelled": 0, "in_progress": 0, "timed_out": counts["timed_out"]}

        logger.warning("vector store batch retrying %d incomplete file(s), attempt %d/%d", len(remaining), attempt + 1, DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS)

    return {"completed": total - len(remaining), "failed": len(remaining), "cancelled": 0, "in_progress": 0, "timed_out": 0}


def get_live_cached_vector_store(
    client: OpenAI,
    repo_root: Path,
    *,
    verbose: bool,
) -> tuple[str | None, str]:
    head_sha = get_repo_head_sha(repo_root)
    cache = load_vector_store_cache(repo_root)
    vector_store_id = cache.get("vector_store_id")
    if not vector_store_id:
        return None, head_sha

    if verbose:
        logger.debug("openai vector_stores.retrieve start vector_store_id=%s", vector_store_id)
    try:
        store = call_with_rate_limit_retry(
            lambda: client.vector_stores.retrieve(vector_store_id),
            what=f"vector store retrieve {vector_store_id}",
            verbose=verbose,
        )
    except Exception:
        if verbose:
            logger.debug("cached vector store missing, dropping cache entry %s", vector_store_id)
        cache["vector_store_id"] = None
        save_vector_store_cache(repo_root, cache)
        return None, head_sha

    if verbose:
        logger.debug("openai vector_stores.retrieve ok vector_store_id=%s", vector_store_id)

    if store.status == "expired":
        if verbose:
            logger.debug("cached vector store expired, dropping cache entry %s", vector_store_id)
        cache["vector_store_id"] = None
        save_vector_store_cache(repo_root, cache)
        return None, head_sha

    return vector_store_id, head_sha


def collect_repo_blob_entries(
    repo_root: Path,
    head_sha: str,
    supported_extensions: set[str],
    *,
    verbose: bool,
) -> tuple[dict[str, tuple[str, str]], int, int]:
    repo_blobs: dict[str, tuple[str, str]] = {}
    skipped_unsupported_count = 0
    duplicate_blob_count = 0

    for relpath, blob_sha in list_head_tree_entries(repo_root, head_sha):
        upload_suffix = get_vector_store_upload_suffix(relpath, supported_extensions)
        if upload_suffix is None:
            skipped_unsupported_count += 1
            if verbose:
                logger.debug("skip vector store file unsupported extension %s", relpath)
            continue
        if blob_sha in repo_blobs:
            duplicate_blob_count += 1
            continue
        repo_blobs[blob_sha] = (relpath, upload_suffix)

    return repo_blobs, skipped_unsupported_count, duplicate_blob_count


def classify_vector_store_sync_entries(
    repo_blobs: dict[str, tuple[str, str]],
    entries: list[VectorStoreBlobEntry],
    *,
    verbose: bool,
) -> tuple[
    dict[str, VectorStoreBlobEntry],
    list[VectorStoreBlobEntry],
    list[VectorStoreBlobEntry],
    list[str],
]:
    ok_entries: dict[str, VectorStoreBlobEntry] = {}
    to_remove_entries: list[VectorStoreBlobEntry] = []
    to_reupload_entries: list[VectorStoreBlobEntry] = []
    seen_blob_shas: set[str] = set()

    for entry in entries:
        blob_sha = entry["blob_sha"]
        status = entry.get("status")
        in_repo = bool(blob_sha) and blob_sha in repo_blobs

        if not blob_sha:
            to_remove_entries.append(entry)
        elif not in_repo:
            to_remove_entries.append(entry)
        elif status == "completed" and blob_sha not in ok_entries:
            ok_entries[blob_sha] = entry
            seen_blob_shas.add(blob_sha)
        elif status == "completed" and blob_sha in ok_entries:
            if verbose:
                logger.warning(
                    "duplicate completed blob_sha in vector store %s %s",
                    blob_sha,
                    entry.get("path", ""),
                )
            to_remove_entries.append(entry)
        else:
            to_reupload_entries.append(entry)
            seen_blob_shas.add(blob_sha)

    to_upload_blob_shas = sorted(set(repo_blobs) - seen_blob_shas)
    return ok_entries, to_remove_entries, to_reupload_entries, to_upload_blob_shas


def delete_vector_store_entries(
    client: OpenAI,
    vector_store_id: str,
    entries: list[VectorStoreBlobEntry],
    *,
    verbose: bool,
) -> int:
    deleted_count = 0
    total_entries = len(entries)

    if total_entries:
        log_progress("vector store delete progress", 0, total_entries)

    for entry in entries:
        delete_vector_store_file(
            client,
            vector_store_id,
            entry["vector_store_file_id"],
            verbose=verbose,
        )
        deleted_count += 1
        if deleted_count == total_entries or deleted_count % 10 == 0:
            log_progress("vector store delete progress", deleted_count, total_entries)

    return deleted_count


def upload_vector_store_blob(
    client: OpenAI,
    repo_root: Path,
    head_sha: str,
    relpath: str,
    blob_sha: str,
    upload_suffix: str,
    *,
    verbose: bool,
) -> str:
    data = git_show_file_bytes(repo_root, head_sha, relpath)
    if data is None:
        raise RuntimeError(f"missing repository blob for {relpath} at {head_sha}")
    if verbose:
        logger.debug("upload vector store file %s %s %s", head_sha, relpath, blob_sha)

    upload_data = build_vector_store_upload_bytes(relpath, blob_sha, data, upload_suffix)
    if verbose and upload_suffix == "txt":
        logger.debug("upload vector store file as txt %s %s %s", head_sha, relpath, blob_sha)

    temp_dir = Path(tempfile.mkdtemp(prefix="openai-vs-batch-"))
    temp_path = build_vector_store_temp_upload_path(temp_dir, relpath, blob_sha, upload_suffix)
    temp_path.write_bytes(upload_data)
    try:
        if verbose:
            logger.debug("openai files.create start %s", relpath)
        file_id = upload_local_file(
            client,
            temp_path,
            what=f"vector store file upload for {relpath}",
            verbose=verbose,
        )
        if verbose:
            logger.debug("openai files.create ok %s %s", relpath, file_id)
        return file_id
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def upload_and_attach_vector_store_blobs(
    client: OpenAI,
    repo_root: Path,
    vector_store_id: str,
    head_sha: str,
    repo_blobs: dict[str, tuple[str, str]],
    blob_shas: list[str],
    *,
    verbose: bool,
) -> tuple[int, int]:
    uploaded_count = 0
    attach_items: list[tuple[str, str, str]] = []
    total_uploads = len(blob_shas)

    if total_uploads:
        log_progress("vector store upload progress", 0, total_uploads)

    for blob_sha in blob_shas:
        relpath, upload_suffix = repo_blobs[blob_sha]
        file_id = upload_vector_store_blob(
            client,
            repo_root,
            head_sha,
            relpath,
            blob_sha,
            upload_suffix,
            verbose=verbose,
        )

        uploaded_count += 1
        attach_items.append((file_id, relpath, blob_sha))
        if uploaded_count == total_uploads or uploaded_count % 10 == 0:
            log_progress("vector store upload progress", uploaded_count, total_uploads)
        if len(attach_items) >= DEFAULT_VECTOR_STORE_FILE_BATCH_SIZE:
            batch_counts = create_vector_store_batch_and_poll(
                client,
                vector_store_id,
                attach_items,
                verbose=verbose,
            )
            logger.info(
                "vector store attach progress: submitted batch of %d, uploaded=%d/%d%s",
                len(attach_items),
                uploaded_count,
                total_uploads,
                " (poll timed out)" if batch_counts.get("timed_out") else "",
            )
            attach_items = []

    if attach_items:
        batch_counts = create_vector_store_batch_and_poll(
            client,
            vector_store_id,
            attach_items,
            verbose=verbose,
        )
        logger.info(
            "vector store attach progress: submitted batch of %d, uploaded=%d/%d%s",
            len(attach_items),
            uploaded_count,
            total_uploads,
            " (poll timed out)" if batch_counts.get("timed_out") else "",
        )

    return uploaded_count, total_uploads


def sync_vector_store_with_repo(
    client: OpenAI,
    repo_root: Path,
    vector_store_id: str,
    head_sha: str,
    *,
    max_retries: int,
    verbose: bool,
) -> dict[str, int]:
    supported_extensions = supported_vector_store_extensions(
        client,
        repo_root,
        verbose=verbose,
    )
    repo_blobs, skipped_unsupported_count, duplicate_blob_count = collect_repo_blob_entries(
        repo_root,
        head_sha,
        supported_extensions,
        verbose=verbose,
    )

    uploaded_total = 0
    removed_total = 0
    sync_passes_used = 0
    retry_passes_used = 0
    remaining_failed = 0
    remaining_missing = 0

    while True:
        all_entries = list_vector_store_blob_entries(
            client,
            vector_store_id,
            verbose=verbose,
        )
        failed_entries = [
            entry for entry in all_entries
            if entry.get("status") != "completed" or not entry.get("blob_sha")
        ]
        ok_entries, to_remove_entries, to_reupload_entries, to_upload_blob_shas = classify_vector_store_sync_entries(
            repo_blobs,
            all_entries,
            verbose=verbose,
        )

        remaining_failed = len(to_reupload_entries)
        remaining_missing = len(to_upload_blob_shas)

        logger.info(
            "vector store sync plan: repo=%d present=%d remove=%d reupload=%d upload=%d attempt=%d",
            len(repo_blobs),
            len(ok_entries),
            len(to_remove_entries),
            len(to_reupload_entries),
            len(to_upload_blob_shas),
            sync_passes_used + 1,
        )

        if failed_entries:
            log_failed_vector_store_blobs(
                failed_entries,
                prefix="vector store failed file",
            )

        if not to_remove_entries and not to_reupload_entries and not to_upload_blob_shas:
            break

        if sync_passes_used > max_retries:
            logger.warning(
                "vector store sync incomplete after %d pass(es): remaining_remove=%d remaining_reupload=%d remaining_upload=%d",
                sync_passes_used,
                len(to_remove_entries),
                len(to_reupload_entries),
                len(to_upload_blob_shas),
            )
            break

        if sync_passes_used > 0:
            retry_passes_used += 1
            logger.warning(
                "retrying vector store sync after pass %d: remaining_remove=%d remaining_reupload=%d remaining_upload=%d",
                sync_passes_used,
                len(to_remove_entries),
                len(to_reupload_entries),
                len(to_upload_blob_shas),
            )

        removed_total += delete_vector_store_entries(
            client,
            vector_store_id,
            to_reupload_entries,
            verbose=verbose,
        )

        blob_shas_to_upload = [entry["blob_sha"] for entry in to_reupload_entries] + to_upload_blob_shas
        uploaded_count, _ = upload_and_attach_vector_store_blobs(
            client,
            repo_root,
            vector_store_id,
            head_sha,
            repo_blobs,
            blob_shas_to_upload,
            verbose=verbose,
        )
        uploaded_total += uploaded_count

        removed_total += delete_vector_store_entries(
            client,
            vector_store_id,
            to_remove_entries,
            verbose=verbose,
        )

        sync_passes_used += 1

    return {
        "repo_files": len(repo_blobs),
        "uploaded_files": uploaded_total,
        "removed_files": removed_total,
        "skipped_unsupported_files": skipped_unsupported_count,
        "duplicate_blob_files": duplicate_blob_count,
        "remaining_failed_files": remaining_failed,
        "remaining_missing_files": remaining_missing,
        "retry_passes_used": retry_passes_used,
    }

def ensure_vector_store_synced(
    client: OpenAI,
    repo_root: Path,
    *,
    expiry_days: int,
    sync_max_retries: int,
    verbose: bool,
) -> tuple[str, str]:
    vector_store_id, head_sha = get_live_cached_vector_store(
        client,
        repo_root,
        verbose=verbose,
    )
    if vector_store_id is None:
        if verbose:
            logger.debug("openai vector_stores.create start")
        store = call_with_rate_limit_retry(
            lambda: client.vector_stores.create(
                name=f"{repo_root.name} repository search index",
                expires_after={"anchor": "last_active_at", "days": max(1, expiry_days)},
            ),
            what="vector_stores.create",
            verbose=verbose,
        )
        vector_store_id = store.id
        if verbose:
            logger.debug("openai vector_stores.create ok vector_store_id=%s", vector_store_id)
        cache = load_vector_store_cache(repo_root)
        cache["vector_store_id"] = vector_store_id
        save_vector_store_cache(repo_root, cache)

    stats = sync_vector_store_with_repo(
        client,
        repo_root,
        vector_store_id,
        head_sha,
        max_retries=sync_max_retries,
        verbose=verbose,
    )

    if verbose:
        logger.debug("vector store repo files: repo=%d uploaded=%d removed=%d missing=%d skipped_unsupported=%d duplicate_blobs=%d", stats["repo_files"], stats["uploaded_files"], stats["removed_files"], stats["remaining_missing_files"], stats["skipped_unsupported_files"], stats["duplicate_blob_files"])

    return vector_store_id, head_sha


def get_live_cached_vector_stores_for_repos(
    client: OpenAI,
    repo_roots: list[Path],
    *,
    verbose: bool,
) -> tuple[list[str], dict[str, str]]:
    vector_store_ids: list[str] = []
    repo_heads: dict[str, str] = {}
    missing_roots: list[Path] = []

    for repo_root in repo_roots:
        if verbose:
            logger.debug("check cached vector store for repo %s", repo_root)
        vector_store_id, head_sha = get_live_cached_vector_store(
            client,
            repo_root,
            verbose=verbose,
        )
        repo_heads[str(repo_root)] = head_sha
        if vector_store_id is None:
            missing_roots.append(repo_root)
            continue
        vector_store_ids.append(vector_store_id)

    if missing_roots:
        missing_text = ", ".join(str(root) for root in missing_roots)
        raise RuntimeError(
            "no cached vector store for: "
            f"{missing_text}; run --prepare-vector-store-only first"
        )

    return vector_store_ids, repo_heads


def ensure_vector_stores_synced_for_repos(
    client: OpenAI,
    repo_roots: list[Path],
    *,
    expiry_days: int,
    sync_max_retries: int,
    verbose: bool,
) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []

    for repo_root in repo_roots:
        if verbose:
            logger.debug("prepare vector store for repo %s", repo_root)
        vector_store_id, head_sha = ensure_vector_store_synced(
            client,
            repo_root,
            expiry_days=expiry_days,
            sync_max_retries=sync_max_retries,
            verbose=verbose,
        )
        prepared.append(
            {
                "repo_root": str(repo_root),
                "vector_store_id": vector_store_id,
                "head_sha": head_sha,
            }
        )

    return prepared


