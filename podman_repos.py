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

Vendor-agnostic repo provisioning for review containers.

Containers always live in the podman host reached over ssh (that host
may be local, a VM, or in the cloud -- it makes no difference here), so
copying a multi-hundred-MB ``.git`` across the wire on *every* review
is exactly what we must avoid. Each repo gets a persistent bare
**mirror on the host**; per review the client ``git push``es just the
head SHA (a thin pack of only the objects the mirror lacks), then the
container is filled **host-locally** (``podman cp`` of the mirror, no
cross-machine copy of the bulk). The first push per repo seeds full
history; later pushes carry only the PR delta. No host filesystem is
bind-mounted into the container.

Why push instead of fetching upstream on the host: one repo
(``all_ffmpeg``) is synthesised on the client via ``git subtree`` and
cannot be reproduced by a plain upstream fetch, so the client -- which
holds the authoritative working tree -- is the only correct source.
"""

from __future__ import annotations

import logging
import random
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from common import dedup_with_suffix, sanitize_repo_name
from git_util import get_repo_head_sha, git_push_refspecs
from podman_host import ContainerHandle, RemoteHost, run_on_remote_host

logger = logging.getLogger(__name__)

DEFAULT_CONTAINER_REPO_ROOT = "/work"
# Remote bare-mirror location (relative to the ssh user's home) and the
# ref each review's head is force-pushed to inside the mirror.
DEFAULT_MIRROR_ROOT = "fairy-mirrors"
MIRROR_REF_PREFIX = "refs/fairy/heads"


@dataclass(frozen=True)
class RepoSpec:
    repo_root: Path
    name: str
    head_sha: str
    container_path: str
    mirror_path: str


def build_repo_specs(
    repo_roots: Sequence[Path],
    *,
    container_root: str = DEFAULT_CONTAINER_REPO_ROOT,
    mirror_root: str = DEFAULT_MIRROR_ROOT,
) -> list[RepoSpec]:
    """Build vendor-neutral specs from host working-tree paths.

    Names are sanitised + suffix-deduped to match the OpenAI-side
    algorithm in ``openai_container.build_container_repo_specs`` so
    operators see the same in-container layout. Each spec carries the
    host bare-mirror path ``<mirror_root>/<name>.git``.
    """
    specs: list[RepoSpec] = []
    used: set[str] = set()
    root = container_root.rstrip("/") or DEFAULT_CONTAINER_REPO_ROOT
    mirror_base = mirror_root.rstrip("/") or DEFAULT_MIRROR_ROOT
    for repo_root in repo_roots:
        name = dedup_with_suffix(sanitize_repo_name(repo_root.name), used)
        t0 = time.monotonic()
        logger.debug("repo head start repo_root=%s", repo_root)
        head_sha = get_repo_head_sha(repo_root)
        logger.debug(
            "repo head ok repo_root=%s head_sha=%s dt=%.3fs",
            repo_root, head_sha, time.monotonic() - t0,
        )
        specs.append(RepoSpec(
            repo_root=repo_root,
            name=name,
            head_sha=head_sha,
            container_path=f"{root}/{name}",
            mirror_path=f"{mirror_base}/{name}.git",
        ))
    return specs


def provision_repos_into_container(
    handle: ContainerHandle,
    specs: Sequence[RepoSpec],
    host: RemoteHost,
) -> None:
    """Populate the container at ``container_path`` for each spec.

    Each repo is synced into its host bare mirror via a thin client
    push, then filled host-locally from that mirror so no bulk repo data
    crosses the wire per review.

    Raises ``RuntimeError`` on any failure -- the caller is expected to
    ``stop_container`` in a ``finally`` block.
    """
    cid = handle.container_id
    for spec in specs:
        logger.info(
            "provisioning repo name=%s head=%s mirror=%s -> %s",
            spec.name, spec.head_sha[:12], spec.mirror_path, spec.container_path,
        )
        t0 = time.monotonic()
        ensure_remote_mirror(host, spec.mirror_path)
        sync_repo_to_mirror(spec, host)
        t_synced = time.monotonic()
        # All podman steps run on the host as the ssh user (same podman
        # the container was started in) so the cp source -- the mirror --
        # stays local to that machine.
        _ssh_podman(host, "exec", cid, "mkdir", "-p", spec.container_path)
        _ssh_podman(host, "cp", spec.mirror_path, f"{cid}:{spec.container_path}/.git",
                    timeout_s=1800.0)
        _ssh_podman(host, "exec", cid, "git", "-C", spec.container_path,
                    "config", "core.bare", "false")
        _ssh_podman(host, "exec", cid, "git", "-C", spec.container_path,
                    "reset", "--hard", spec.head_sha)
        done = time.monotonic()
        logger.info(
            "provisioned repo name=%s sync=%.3fs fill=%.3fs total=%.3fs",
            spec.name, t_synced - t0, done - t_synced, done - t0,
        )


def ensure_remote_mirror(
    host: RemoteHost, mirror_path: str, *, timeout_s: float = 120.0,
) -> None:
    """Idempotently create the bare mirror ``mirror_path`` on ``host``.

    ``git init --bare`` is a no-op on an existing repo, so this is safe
    to call before every push. Provisioning seeds mirrors up front; this
    keeps per-review provisioning self-sufficient if it was not.
    """
    parent = mirror_path.rsplit("/", 1)[0] if "/" in mirror_path else "."
    _check_remote(run_on_remote_host(host, "mkdir", "-p", parent, timeout_s=timeout_s),
                  f"mkdir -p {parent}", host)
    _check_remote(run_on_remote_host(host, "git", "init", "--bare", "--quiet", mirror_path,
                                     timeout_s=timeout_s),
                  f"git init --bare {mirror_path}", host)


def sync_repo_to_mirror(
    spec: RepoSpec, host: RemoteHost, *, timeout_s: float = 600.0, attempts: int = 3,
) -> None:
    """Force-push ``spec.head_sha`` and ALL client refs into the mirror.

    The full ``refs/*`` refspec carries every ref the client repo has --
    for ffmpeg that includes the thousands of ``refs/remotes/fforge/pr/*``
    PR heads -- so the container checkout can resolve any PR, not just
    the review head. Only objects the mirror lacks are sent (thin pack):
    cheap after the one-time full seed. Concurrent reviews sync the same
    repo, and the loser's push fails with a transient "cannot lock ref";
    retry up to ``attempts`` times with a short random delay (a retry
    pushing SHAs the winner already placed succeeds as a no-op).
    """
    ssh_command = shlex.join(
        ["ssh", *host.ssh_opts, *(["-i", host.identity] if host.identity else [])]
    )
    remote_url = f"{host.ssh_dest}:{spec.mirror_path}"
    dest_ref = f"{MIRROR_REF_PREFIX}/{spec.name}"
    refspecs = [f"{spec.head_sha}:{dest_ref}", "refs/*:refs/*"]
    logger.info(
        "syncing repo (push) name=%s head=%s -> %s %s + all refs",
        spec.name, spec.head_sha[:12], remote_url, dest_ref,
    )
    t0 = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            git_push_refspecs(
                spec.repo_root, remote_url, refspecs,
                ssh_command=ssh_command, timeout_s=timeout_s,
            )
            logger.info(
                "synced repo name=%s head=%s dt=%.3fs attempts=%d",
                spec.name, spec.head_sha[:12], time.monotonic() - t0, attempt,
            )
            return
        except RuntimeError as exc:
            if attempt == attempts:
                raise
            delay = random.uniform(0.5, 3.0)
            logger.info(
                "mirror push failed (attempt %d/%d) name=%s: %s; retrying in %.1fs",
                attempt, attempts, spec.name,
                str(exc).replace("\n", " "), delay,
            )
            time.sleep(delay)


def _ssh_podman(host: RemoteHost, *args: str, timeout_s: float = 120.0) -> None:
    _check_remote(run_on_remote_host(host, "podman", *args, timeout_s=timeout_s),
                  "podman " + " ".join(args), host)


def _check_remote(result, what: str, host: RemoteHost) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"remote-local {what!r} on {host.ssh_dest} failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
