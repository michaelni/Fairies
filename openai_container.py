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

Helpers for managing OpenAI containers attached to the review wrapper."""

from __future__ import annotations

import json
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from openai import BadRequestError, NotFoundError, OpenAI

import openai_container_pool
from common import atomic_write_text, dedup_with_suffix, sanitize_repo_name
from git_util import get_repo_head_sha
from openai_common import (
    JsonObject,
    ResponseKwargs,
    _obj_get,
    call_with_rate_limit_retry,
    delete_uploaded_file,
    extract_response_text,
    openai_file_exists,
    upload_local_file,
    upload_text_file,
)

logger = logging.getLogger(__name__)


ContainerRepoCache: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class ContainerRepoSpec:
    repo_root: Path
    repo_key: str
    repo_name: str
    head_sha: str
    archive_name: str
    mounted_path: str


DEFAULT_CONTAINER_EXPIRY_MINUTES = 20
DEFAULT_CONTAINER_MEMORY_LIMIT = "1g"
DEFAULT_CONTAINER_REPOS_ROOT = "/mnt/data/repos"
DEFAULT_CONTAINER_BOOTSTRAP_MAX_OUTPUT_TOKENS = 4_000
DEFAULT_CONTAINER_BOOTSTRAP_MODEL = "gpt-5.4-nano"
CONTAINER_REPO_CACHE_FILENAME = ".openai_container_repo_cache.json"
# Bump this when the bootstrap script logic changes in a way that makes
# older containers non-interchangeable with newer ones. Included in the
# pool state hash so old pool entries are naturally orphaned.
CONTAINER_BOOTSTRAP_VERSION = "1"


@dataclass
class ContainerLease:
    """Exclusive handle to a ready OpenAI container.

    The caller owns ``container_id`` until ``release`` is called. When
    ``pool_dir`` is ``None`` the lease is not pool-managed (caller
    supplied ``explicit_container_id``) and ``release`` is a no-op.
    """

    container_id: str
    specs: list[ContainerRepoSpec]
    pool_dir: Path | None

    def release(self, *, healthy: bool) -> bool:
        if self.pool_dir is None:
            return False
        return openai_container_pool.release_container(
            self.pool_dir, self.container_id, healthy=healthy,
        )


def compute_container_pool_state_hash(
    specs: list[ContainerRepoSpec],
    *,
    container_repos_root: str,
    memory_limit: str,
) -> str:
    """Hash every input that determines container contents.

    Two runs whose hashes match must be safe to interchange containers.
    Do NOT include per-run values (expiry_minutes, timestamps).
    """
    state = {
        "bootstrap_version": CONTAINER_BOOTSTRAP_VERSION,
        "container_repos_root": container_repos_root,
        "memory_limit": memory_limit,
        "repos": sorted(
            (
                {
                    "repo_key": spec.repo_key,
                    "repo_name": spec.repo_name,
                    "head_sha": spec.head_sha,
                }
                for spec in specs
            ),
            key=lambda entry: entry["repo_key"],
        ),
    }
    return openai_container_pool.compute_container_state_hash(state)


def normalize_container_repo_cache(data: object) -> ContainerRepoCache:
    if not isinstance(data, dict):
        return {
            "container_id": None,
            "container_repos_root": DEFAULT_CONTAINER_REPOS_ROOT,
            "repo_heads": {},
            "repo_entries": {},
        }

    container_id = data.get("container_id")
    if not isinstance(container_id, str) or not container_id:
        container_id = None

    container_repos_root = data.get("container_repos_root")
    if not isinstance(container_repos_root, str) or not container_repos_root:
        container_repos_root = DEFAULT_CONTAINER_REPOS_ROOT

    repo_heads_raw = data.get("repo_heads")
    repo_heads: dict[str, str] = {}
    if isinstance(repo_heads_raw, dict):
        for repo_key, head_sha in repo_heads_raw.items():
            if isinstance(repo_key, str) and repo_key and isinstance(head_sha, str) and head_sha:
                repo_heads[repo_key] = head_sha

    repo_entries_raw = data.get("repo_entries")
    repo_entries: dict[str, JsonObject] = {}
    if isinstance(repo_entries_raw, dict):
        for repo_key, raw_entry in repo_entries_raw.items():
            if not isinstance(repo_key, str) or not repo_key or not isinstance(raw_entry, dict):
                continue
            entry: JsonObject = {}
            for field in (
                "repo_name",
                "head_sha",
                "archive_name",
                "uploaded_file_id",
                "container_file_path",
                "mounted_path",
            ):
                value = raw_entry.get(field)
                if isinstance(value, str) and value:
                    entry[field] = value
            repo_entries[repo_key] = entry

    return {
        "container_id": container_id,
        "container_repos_root": container_repos_root,
        "repo_heads": repo_heads,
        "repo_entries": repo_entries,
    }


def load_container_repo_cache(cache_root: Path) -> ContainerRepoCache:
    cache_path = cache_root / CONTAINER_REPO_CACHE_FILENAME
    if not cache_path.exists():
        return normalize_container_repo_cache({})
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("%s", exc)
        return normalize_container_repo_cache({})
    return normalize_container_repo_cache(data)


# Fragments of the nested OpenAI error ``message`` field that indicate the
# container we attached to is in a state we cannot use. In every observed
# case the correct recovery is the same: drop the container from the pool
# and let the caller retry against a freshly provisioned one.
CONTAINER_UNHEALTHY_MESSAGE_FRAGMENTS: tuple[str, ...] = (
    "container is expired",
    "container has expired",
    "container is not running",
)


def is_container_unhealthy_error(exc: BaseException) -> bool:
    """True if ``exc`` is an OpenAI error saying the attached container
    is in an unusable state (expired, stopped, etc.).

    Example errors seen in the wild:
        BadRequestError: Error code: 400 - {'error': {'message': 'Container is expired.', ...}}
        BadRequestError: Error code: 400 - {'error': {'message': 'Container is not running.', ...}}
        NotFoundError:   Error code: 404 - {'error': {'message': 'Container has expired.', ...}}

    The same logical failure (the container is gone) surfaces as both a
    400 ``BadRequestError`` and a 404 ``NotFoundError`` with either an
    "is expired" or "has expired" phrasing, so both types and both
    phrasings are matched.

    The SDK builds ``exc.message`` as ``f"Error code: {status} - {body}"``
    (see ``openai._base_client._make_status_error_from_response``), so
    ``str(exc)`` always contains the stringified body including the nested
    ``'message': '...'`` fragment. Substring checks against that are
    sufficient; no need to dig into ``exc.body``.

    The retrieve-time check in :func:`get_live_container_id` cannot prevent
    this fully because a container can transition to a bad state any time
    between retrieve and use (and the API does not always reject the
    retrieve immediately).
    """
    if not isinstance(exc, (BadRequestError, NotFoundError)):
        return False
    text = str(exc).lower()
    return any(fragment in text for fragment in CONTAINER_UNHEALTHY_MESSAGE_FRAGMENTS)


def invalidate_container_repo_caches_for_roots(repo_roots: list[Path], *, verbose: bool) -> int:
    """Drop ``container_id`` from every per-repo container cache file.

    Used when the API tells us the cached container is expired, so the
    next wrapper invocation builds a fresh container instead of reusing
    a dead one. Returns the number of caches that actually changed.
    """
    invalidated = 0
    for repo_root in repo_roots:
        per_repo_cache = load_container_repo_cache(repo_root)
        previous_id = per_repo_cache.get("container_id")
        if not (isinstance(previous_id, str) and previous_id):
            continue
        per_repo_cache["container_id"] = None
        save_container_repo_cache(repo_root, per_repo_cache)
        invalidated += 1
        if verbose:
            logger.debug(
                "container repo cache container_id invalidated repo_root=%s previous_container_id=%s",
                repo_root,
                previous_id,
            )
    return invalidated


def save_container_repo_cache(cache_root: Path, cache: ContainerRepoCache) -> None:
    cache_path = cache_root / CONTAINER_REPO_CACHE_FILENAME
    atomic_write_text(
        cache_path,
        json.dumps(normalize_container_repo_cache(cache), indent=2, sort_keys=True),
    )


def build_container_archive_name(repo_name: str, head_sha: str) -> str:
    return f"{sanitize_repo_name(repo_name)}-{head_sha[:16]}.tar"


def build_container_repo_specs(
    repo_roots: list[Path],
    *,
    container_repos_root: str,
    verbose: bool,
) -> list[ContainerRepoSpec]:
    specs: list[ContainerRepoSpec] = []
    used_names: set[str] = set()
    root_prefix = container_repos_root.rstrip("/") or DEFAULT_CONTAINER_REPOS_ROOT

    for repo_root in repo_roots:
        repo_name = dedup_with_suffix(sanitize_repo_name(repo_root.name), used_names)

        started = time.monotonic()
        if verbose:
            logger.debug("container repo head start repo_root=%s", repo_root)
        head_sha = get_repo_head_sha(repo_root)
        if verbose:
            logger.debug(
                "container repo head ok repo_root=%s head_sha=%s dt=%.3fs",
                repo_root,
                head_sha,
                time.monotonic() - started,
            )
        repo_key = str(repo_root.resolve())
        specs.append(
            ContainerRepoSpec(
                repo_root=repo_root,
                repo_key=repo_key,
                repo_name=repo_name,
                head_sha=head_sha,
                archive_name=build_container_archive_name(repo_name, head_sha),
                mounted_path=f"{root_prefix}/{repo_name}",
            )
        )

    return specs


def build_container_repo_archive(temp_dir: Path, spec: ContainerRepoSpec, *, verbose: bool) -> Path:
    archive_path = temp_dir / spec.archive_name
    git_dir = spec.repo_root / ".git"
    started = time.monotonic()
    if verbose:
        logger.debug(
            "container repo archive build start repo_root=%s git_dir=%s archive=%s",
            spec.repo_root,
            git_dir,
            archive_path,
        )
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(git_dir, arcname=f"{spec.repo_name}/.git")
    if verbose:
        logger.debug(
            "container repo archive build ok repo_root=%s archive=%s bytes=%d dt=%.3fs",
            spec.repo_root,
            archive_path,
            archive_path.stat().st_size,
            time.monotonic() - started,
        )
    return archive_path


def build_container_cache_repo_entry(spec: ContainerRepoSpec, existing_entry: object) -> JsonObject:
    entry = existing_entry if isinstance(existing_entry, dict) else {}
    normalized: JsonObject = {
        "repo_name": spec.repo_name,
        "head_sha": spec.head_sha,
        "archive_name": spec.archive_name,
        "mounted_path": spec.mounted_path,
    }
    for field in ("uploaded_file_id", "container_file_path"):
        value = entry.get(field) if isinstance(entry, dict) else None
        if isinstance(value, str) and value:
            normalized[field] = value
    return normalized


def ensure_uploaded_container_repo_archives(
    client: OpenAI,
    specs: list[ContainerRepoSpec],
    cache: ContainerRepoCache,
    *,
    verbose: bool,
) -> dict[str, JsonObject]:
    existing_entries = cache.get("repo_entries")
    entries_by_repo: dict[str, JsonObject] = existing_entries if isinstance(existing_entries, dict) else {}
    next_entries: dict[str, JsonObject] = {}
    next_heads: dict[str, str] = {}

    for spec in specs:
        existing_entry = entries_by_repo.get(spec.repo_key)
        entry = build_container_cache_repo_entry(spec, existing_entry)
        cached_head_sha = existing_entry.get("head_sha") if isinstance(existing_entry, dict) else None
        cached_archive_name = existing_entry.get("archive_name") if isinstance(existing_entry, dict) else None
        if cached_head_sha != spec.head_sha or cached_archive_name != spec.archive_name:
            entry.pop("uploaded_file_id", None)
            entry.pop("container_file_path", None)

        uploaded_file_id = entry.get("uploaded_file_id")
        if not isinstance(uploaded_file_id, str) or not uploaded_file_id or not openai_file_exists(client, uploaded_file_id, verbose=verbose):
            if verbose:
                logger.debug(
                    "container repo archive upload required repo_root=%s repo_name=%s cached_head_sha=%s head_sha=%s file_id=%s",
                    spec.repo_root,
                    spec.repo_name,
                    cached_head_sha or "-",
                    spec.head_sha,
                    uploaded_file_id if isinstance(uploaded_file_id, str) and uploaded_file_id else "-",
                )
            with tempfile.TemporaryDirectory(prefix="openai-container-repo-archive-") as temp_dir_name:
                archive_path = build_container_repo_archive(Path(temp_dir_name), spec, verbose=verbose)
                uploaded_file_id = upload_local_file(
                    client,
                    archive_path,
                    what=f"container repo archive upload for {spec.repo_name}",
                    verbose=verbose,
                )
            entry["uploaded_file_id"] = uploaded_file_id
            entry.pop("container_file_path", None)
        elif verbose:
            logger.debug(
                "container repo archive upload cached repo_root=%s repo_name=%s file_id=%s",
                spec.repo_root,
                spec.repo_name,
                uploaded_file_id,
            )

        next_entries[spec.repo_key] = entry
        next_heads[spec.repo_key] = spec.head_sha

    cache["repo_entries"] = next_entries
    cache["repo_heads"] = next_heads
    repos_root = specs[0].mounted_path.rsplit("/", 1)[0] if specs else DEFAULT_CONTAINER_REPOS_ROOT
    cache["container_repos_root"] = repos_root
    return next_entries


def list_container_files(
    client: OpenAI,
    container_id: str,
    *,
    verbose: bool,
) -> list[object]:
    items: list[object] = []
    after: str | None = None

    while True:
        page_started = time.monotonic()
        if verbose:
            logger.debug("openai containers.files.list start container_id=%s after=%s", container_id, after or "-")

        def fetch_list() -> object:
            kwargs: dict[str, object] = {"container_id": container_id, "limit": 100, "order": "asc"}
            if after is not None:
                kwargs["after"] = after
            return client.containers.files.list(**kwargs)

        page = call_with_rate_limit_retry(
            fetch_list,
            what=f"container files list for {container_id}",
            verbose=verbose,
        )
        page_items = _obj_get(page, "data", [])
        if not isinstance(page_items, list):
            page_items = []
        items.extend(page_items)
        if verbose:
            logger.debug(
                "openai containers.files.list ok container_id=%s after=%s page_items=%d total_items=%d dt=%.3fs",
                container_id,
                after or "-",
                len(page_items),
                len(items),
                time.monotonic() - page_started,
            )

        if not _obj_get(page, "has_more", False):
            break
        last_id = _obj_get(page, "last_id", None)
        if not isinstance(last_id, str) or not last_id:
            break
        after = last_id

    return items


def get_live_container_id(
    client: OpenAI,
    container_id: str,
    *,
    verbose: bool,
    strict: bool,
) -> str | None:
    started = time.monotonic()
    if verbose:
        logger.debug("openai containers.retrieve start container_id=%s strict=%s", container_id, strict)
    try:
        container = call_with_rate_limit_retry(
            lambda: client.containers.retrieve(container_id),
            what=f"container retrieve {container_id}",
            verbose=verbose,
        )
    except Exception:
        if strict:
            raise
        if verbose:
            logger.debug("cached container is not retrievable: %s", container_id)
        return None

    status = _obj_get(container, "status", None)
    if status in {"expired", "deleted"}:
        if strict:
            raise RuntimeError(f"container {container_id} is {status}")
        return None
    live_id = _obj_get(container, "id", None)
    if not isinstance(live_id, str) or not live_id:
        if strict:
            raise RuntimeError(f"container retrieve did not return a usable id for {container_id}")
        return None
    if verbose:
        logger.debug(
            "openai containers.retrieve ok container_id=%s status=%s dt=%.3fs",
            live_id,
            status or "-",
            time.monotonic() - started,
        )
    return live_id


def create_fresh_openai_container(
    client: OpenAI,
    specs: list[ContainerRepoSpec],
    *,
    memory_limit: str,
    expiry_minutes: int,
    verbose: bool,
) -> str:
    """Create a brand-new OpenAI container. Caller owns the returned id.

    Reuse of existing containers is now handled by the container pool
    (see :func:`ensure_openai_container_repos_ready`); this helper
    simply provisions a fresh one each time.
    """
    create_started = time.monotonic()
    if verbose:
        logger.debug(
            "openai containers.create start repo_keys=%s memory_limit=%s expiry_minutes=%d",
            json.dumps([spec.repo_key for spec in specs], ensure_ascii=False),
            memory_limit,
            max(1, expiry_minutes),
        )
    container = call_with_rate_limit_retry(
        lambda: client.containers.create(
            name="repo container",
            memory_limit=memory_limit,
            expires_after={"anchor": "last_active_at", "minutes": max(1, expiry_minutes)},
        ),
        what="containers.create",
        verbose=verbose,
    )
    container_id = _obj_get(container, "id", None)
    if not isinstance(container_id, str) or not container_id:
        raise RuntimeError("container creation did not return a usable id")
    if verbose:
        logger.debug(
            "openai containers.create ok container_id=%s dt=%.3fs",
            container_id,
            time.monotonic() - create_started,
        )
    return container_id


def attach_container_repo_archives(
    client: OpenAI,
    container_id: str,
    specs: list[ContainerRepoSpec],
    repo_entries: dict[str, JsonObject],
    *,
    verbose: bool,
) -> None:
    existing_paths = {
        path
        for item in list_container_files(client, container_id, verbose=verbose)
        for path in [_obj_get(item, "path", None)]
        if isinstance(path, str) and path
    }

    for spec in specs:
        entry = repo_entries.get(spec.repo_key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"missing cache entry for repository {spec.repo_root}")

        container_file_path = entry.get("container_file_path")
        if isinstance(container_file_path, str) and container_file_path in existing_paths:
            if verbose:
                logger.debug(
                    "container archive already attached container_id=%s repo_name=%s path=%s",
                    container_id,
                    spec.repo_name,
                    container_file_path,
                )
            continue

        uploaded_file_id = entry.get("uploaded_file_id")
        if not isinstance(uploaded_file_id, str) or not uploaded_file_id:
            raise RuntimeError(f"missing uploaded file id for repository {spec.repo_root}")

        started = time.monotonic()
        if verbose:
            logger.debug(
                "openai containers.files.create start container_id=%s repo_name=%s file_id=%s",
                container_id,
                spec.repo_name,
                uploaded_file_id,
            )
        created = call_with_rate_limit_retry(
            lambda: client.containers.files.create(
                container_id=container_id,
                file_id=uploaded_file_id,
            ),
            what=f"container file create for {spec.repo_name}",
            verbose=verbose,
        )
        container_file_path = _obj_get(created, "path", None)
        if not isinstance(container_file_path, str) or not container_file_path:
            raise RuntimeError(f"container file create did not return a usable path for {spec.repo_name}")
        entry["container_file_path"] = container_file_path
        if verbose:
            logger.debug(
                "openai containers.files.create ok container_id=%s repo_name=%s path=%s dt=%.3fs",
                container_id,
                spec.repo_name,
                container_file_path,
                time.monotonic() - started,
            )


def upload_container_text_file(
    client: OpenAI,
    container_id: str,
    *,
    filename: str,
    text: str,
    verbose: bool,
) -> str:
    if verbose:
        logger.debug(
            "upload container text file start container_id=%s filename=%s bytes=%d",
            container_id,
            filename,
            len(text.encode("utf-8")),
        )
    uploaded_file_id = upload_text_file(
        client,
        filename=filename,
        text=text,
        verbose=verbose,
    )
    try:
        created = call_with_rate_limit_retry(
            lambda: client.containers.files.create(
                container_id=container_id,
                file_id=uploaded_file_id,
            ),
            what=f"container file create for {filename}",
            verbose=verbose,
        )
    finally:
        delete_uploaded_file(client, uploaded_file_id, verbose=verbose)

    container_file_path = _obj_get(created, "path", None)
    if not isinstance(container_file_path, str) or not container_file_path:
        raise RuntimeError(f"container file create did not return a usable path for {filename}")
    if verbose:
        logger.debug(
            "upload container text file ok container_id=%s filename=%s path=%s",
            container_id,
            filename,
            container_file_path,
        )
    return container_file_path


def build_container_repo_bootstrap_command(
    script_path: str,
) -> str:
    return f"python3 {json.dumps(script_path)}"


def build_container_repo_bootstrap_script(
    specs: list[ContainerRepoSpec],
    repo_entries: dict[str, JsonObject],
    *,
    container_repos_root: str,
) -> str:
    manifest: list[dict[str, str]] = []
    for spec in specs:
        entry = repo_entries.get(spec.repo_key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"missing cache entry for repository {spec.repo_root}")
        archive_path = entry.get("container_file_path")
        if not isinstance(archive_path, str) or not archive_path:
            raise RuntimeError(f"missing container archive path for repository {spec.repo_root}")
        manifest.append(
            {
                "repo_name": spec.repo_name,
                "head_sha": spec.head_sha,
                "archive_path": archive_path,
                "mounted_path": spec.mounted_path,
            }
        )

    manifest_json = json.dumps(manifest, ensure_ascii=False)
    repos_root_json = json.dumps(container_repos_root)
    return (
        "import json\n"
        "import pathlib\n"
        "import shutil\n"
        "import tarfile\n"
        "\n"
        f"MANIFEST = json.loads({json.dumps(manifest_json)})\n"
        f"REPOS_ROOT = pathlib.Path({repos_root_json})\n"
        "\n"
        "def main() -> int:\n"
        "    REPOS_ROOT.mkdir(parents=True, exist_ok=True)\n"
        "    for item in MANIFEST:\n"
        "        target = pathlib.Path(item['mounted_path'])\n"
        "        marker = target / '.openai_repo_head'\n"
        "        if marker.exists() and marker.read_text(encoding='utf-8').strip() == item['head_sha']:\n"
        "            print(f\"reuse {item['repo_name']} {target}\")\n"
        "            continue\n"
        "        shutil.rmtree(target, ignore_errors=True)\n"
        "        with tarfile.open(item['archive_path'], 'r:') as archive:\n"
        "            archive.extractall(REPOS_ROOT)\n"
        "        target.mkdir(parents=True, exist_ok=True)\n"
        "        marker.write_text(item['head_sha'] + '\\n', encoding='utf-8')\n"
        "        print(f\"ready {item['repo_name']} {target}\")\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )


def bootstrap_openai_container_repos(
    client: OpenAI,
    *,
    container_id: str,
    script_path: str,
    bootstrap_manifest: list[dict[str, str]],
    verbose: bool,
) -> None:
    command = build_container_repo_bootstrap_command(script_path)
    if verbose:
        logger.debug(
            "container bootstrap start model=%s container_id=%s script_path=%s command=%s repos=%s",
            DEFAULT_CONTAINER_BOOTSTRAP_MODEL,
            container_id,
            script_path,
            command,
            json.dumps(bootstrap_manifest, ensure_ascii=False, sort_keys=True),
        )
    response_kwargs: ResponseKwargs = {
        "model": DEFAULT_CONTAINER_BOOTSTRAP_MODEL,
        "tool_choice": "required",
        "max_output_tokens": DEFAULT_CONTAINER_BOOTSTRAP_MAX_OUTPUT_TOKENS,
        "tools": [
            {
                "type": "shell",
                "environment": {
                    "type": "container_reference",
                    "container_id": container_id,
                },
            }
        ],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Prepare the repository archives inside the shared container.\n"
                            "Run the following shell command exactly once, do not modify it, and reply exactly READY after it finishes successfully.\n\n"
                            f"{command}"
                        ),
                    }
                ],
            }
        ],
    }
    try:
        response = call_with_rate_limit_retry(
            lambda: client.responses.create(**response_kwargs),
            what="container repo bootstrap responses.create",
            verbose=verbose,
        )
    except Exception as exc:
        logger.error(
            "container bootstrap failed model=%s container_id=%s script_path=%s command=%s error=%s",
            DEFAULT_CONTAINER_BOOTSTRAP_MODEL,
            container_id,
            script_path,
            command,
            str(exc).replace("\n", " "),
        )
        raise
    raw_text = extract_response_text(
        response,
        response_kwargs=response_kwargs,
        debug_dir=".openai_debug",
        verbose=verbose,
    )
    if verbose:
        logger.debug(
            "container bootstrap response model=%s container_id=%s script_path=%s text=%s",
            DEFAULT_CONTAINER_BOOTSTRAP_MODEL,
            container_id,
            script_path,
            raw_text.replace("\n", "\\n"),
        )
    if raw_text.strip() != "READY":
        raise RuntimeError(f"container bootstrap did not confirm readiness: {raw_text}")


def ensure_openai_container_repos_ready(
    client: OpenAI,
    repo_roots: list[Path],
    *,
    explicit_container_id: str | None,
    container_repos_root: str,
    expiry_minutes: int,
    memory_limit: str,
    verbose: bool,
    pool_root: Path | None = None,
) -> ContainerLease:
    """Return a :class:`ContainerLease` owning a ready container.

    * If ``explicit_container_id`` is set, verify it is live and return
      a lease with ``pool_dir=None`` (the caller picked this id; the
      pool is not involved).
    * Otherwise, claim entries from the pool keyed by
      :func:`compute_container_pool_state_hash`, verifying each with
      :func:`get_live_container_id` before returning it. Stale entries
      (expired/deleted) are unlinked and skipped; the loop terminates
      either at the first live entry or once the pool is empty, in
      which case we provision a fresh container inline (upload archives
      + create + attach + bootstrap).
    """
    specs = build_container_repo_specs(
        repo_roots,
        container_repos_root=container_repos_root,
        verbose=verbose,
    )

    if explicit_container_id:
        live_container_id = get_live_container_id(
            client, explicit_container_id, verbose=verbose, strict=True,
        )
        if live_container_id is None:
            raise RuntimeError(f"container {explicit_container_id} is not usable")
        return ContainerLease(
            container_id=live_container_id, specs=specs, pool_dir=None,
        )

    effective_pool_root = pool_root if pool_root is not None else openai_container_pool.DEFAULT_POOL_ROOT
    state_hash = compute_container_pool_state_hash(
        specs, container_repos_root=container_repos_root, memory_limit=memory_limit,
    )
    pool_dir = openai_container_pool.pool_dir_for(effective_pool_root, state_hash)
    logger.debug("container pool lookup pool_dir=%s specs=%d", pool_dir, len(specs))

    while True:
        claimed = openai_container_pool.try_claim_container(pool_dir)
        if claimed is None:
            break
        if get_live_container_id(client, claimed, verbose=verbose, strict=False):
            return ContainerLease(
                container_id=claimed, specs=specs, pool_dir=pool_dir,
            )
        logger.warning(
            "container pool stale entry discarded pool_dir=%s container_id=%s",
            pool_dir, claimed,
        )

    # Pool empty: provision a fresh container.
    existing_entries: dict[str, JsonObject] = {}
    for spec in specs:
        cache = load_container_repo_cache(spec.repo_root)
        cached_entries = cache.get("repo_entries")
        if isinstance(cached_entries, dict):
            cached_entry = cached_entries.get(spec.repo_key)
            if isinstance(cached_entry, dict):
                existing_entries[spec.repo_key] = dict(cached_entry)

    merged_cache = normalize_container_repo_cache(
        {
            "container_repos_root": container_repos_root,
            "repo_heads": {spec.repo_key: spec.head_sha for spec in specs},
            "repo_entries": existing_entries,
        }
    )
    repo_entries = ensure_uploaded_container_repo_archives(
        client, specs, merged_cache, verbose=verbose,
    )
    for spec in specs:
        per_repo_cache = normalize_container_repo_cache({})
        per_repo_cache["container_id"] = None
        per_repo_cache["container_repos_root"] = container_repos_root
        per_repo_cache["repo_heads"] = {spec.repo_key: spec.head_sha}
        if spec.repo_key in repo_entries:
            per_repo_cache["repo_entries"] = {spec.repo_key: repo_entries[spec.repo_key]}
        save_container_repo_cache(spec.repo_root, per_repo_cache)

    container_id = create_fresh_openai_container(
        client, specs, memory_limit=memory_limit, expiry_minutes=expiry_minutes, verbose=verbose,
    )
    attach_container_repo_archives(client, container_id, specs, repo_entries, verbose=verbose)
    script_path = upload_container_text_file(
        client,
        container_id,
        filename="openai_container_repo_extract.py",
        text=build_container_repo_bootstrap_script(
            specs, repo_entries, container_repos_root=container_repos_root,
        ),
        verbose=verbose,
    )
    bootstrap_manifest = [
        {
            "repo_name": spec.repo_name,
            "head_sha": spec.head_sha,
            "archive_path": str(repo_entries.get(spec.repo_key, {}).get("container_file_path", "")),
            "mounted_path": spec.mounted_path,
        }
        for spec in specs
    ]
    bootstrap_openai_container_repos(
        client,
        container_id=container_id,
        script_path=script_path,
        bootstrap_manifest=bootstrap_manifest,
        verbose=verbose,
    )
    return ContainerLease(
        container_id=container_id, specs=specs, pool_dir=pool_dir,
    )
