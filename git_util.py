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

Small ``git`` subprocess helpers shared across the project.

Thin wrappers around ``git show`` / ``git rev-parse HEAD`` /
``git ls-tree`` used wherever we need to read from a local checkout
without bringing in a full git library.

This is a leaf module: it has no in-repo dependencies.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

# Remote names a deployment's checkouts use for the forge, in probe
# order: fairy's own mirror setups name it fforge, plain clones origin.
FORGE_REMOTES = ("fforge", "origin")


def git_show_file_bytes(repo_root: Path, revision: str, relpath: str) -> bytes | None:
    spec = f"{revision}:{relpath}"
    cp = subprocess.run(
        ["git", "-C", str(repo_root), "show", spec],
        check=False,
        text=False,
        capture_output=True,
    )
    if cp.returncode != 0:
        return None
    return cp.stdout


def git_show_file(repo_root: Path, revision: str, relpath: str) -> str | None:
    data = git_show_file_bytes(repo_root, revision, relpath)
    if data is None:
        return None
    return data.decode("utf-8", errors="replace")


def get_repo_head_sha(repo_root: Path) -> str:
    return git_rev_parse(repo_root, "HEAD")


def git_rev_parse(repo_root: Path, ref: str) -> str:
    """Resolve ``ref`` to its full SHA in ``repo_root``."""
    cp = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", ref],
        check=False, text=True, capture_output=True,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        raise RuntimeError(
            f"git rev-parse {ref!r} in {repo_root} failed: "
            f"{cp.stderr.strip() or 'empty output'}"
        )
    return cp.stdout.strip()


def git_resolve_first(repo_root: Path, refs: Sequence[str]) -> str | None:
    """The SHA of the first ref in ``refs`` that resolves in
    ``repo_root``; None when none does."""
    for ref in refs:
        try:
            return git_rev_parse(repo_root, ref)
        except RuntimeError:
            continue
    return None


def git_merge_base(repo_root: Path, sha_a: str, sha_b: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", sha_a, sha_b],
        check=False, text=True, capture_output=True,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        raise RuntimeError(
            f"git merge-base {sha_a} {sha_b} in {repo_root} failed: "
            f"{cp.stderr.strip() or 'no common ancestor'}"
        )
    return cp.stdout.strip()


def git_merge_tree(repo_root: Path, sha_a: str, sha_b: str) -> str | None:
    """Tree OID of merging ``sha_a`` and ``sha_b`` without a worktree
    (``git merge-tree --write-tree``); None when the merge conflicts or
    merge-tree is unavailable (git < 2.38)."""
    cp = subprocess.run(
        ["git", "-C", str(repo_root), "merge-tree", "--write-tree", sha_a, sha_b],
        check=False, text=True, capture_output=True,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    return cp.stdout.splitlines()[0].strip()


def _git_stdout(repo_root: Path, *args: str) -> bytes:
    """stdout of ``git <args>`` in ``repo_root``; RuntimeError on failure."""
    cp = subprocess.run(["git", "-C", str(repo_root), *args],
                        capture_output=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} in {repo_root} failed: "
            f"{cp.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return cp.stdout


def git_format_patch_series(
    repo_root: Path, base_sha: str, head_sha: str,
) -> bytes:
    """Return ``git format-patch base..head --stdout`` for ``repo_root``.

    ``--diff-algorithm=histogram --minimal`` produces a more stable
    diff than the Myers default; the output is an mbox of one
    ``From <sha>`` per commit, which matches what Forgejo's ``.patch``
    URL emits (modulo the trailing git-version stamp).
    """
    return _git_stdout(repo_root, "format-patch", "--diff-algorithm=histogram",
                       "--minimal", "--stdout", f"{base_sha}..{head_sha}")


def git_log_patches(repo_root: Path, base_sha: str, head_sha: str) -> bytes:
    """``git log --patch --reverse base..head`` -- every commit of the
    range oldest first, under its real hash and author. Unlike
    format-patch output it includes merge commits (header and message;
    git prints no diff for a merge)."""
    return _git_stdout(repo_root, "log", "--patch", "--reverse",
                       "--no-decorate", "--diff-algorithm=histogram",
                       "--minimal", f"{base_sha}..{head_sha}")


def git_diff(repo_root: Path, base_sha: str, head_sha: str) -> bytes:
    """``git diff base head`` -- the accumulated change merging
    ``head_sha`` would introduce onto ``base_sha``, with the same
    histogram/minimal settings as git_format_patch_series."""
    return _git_stdout(repo_root, "diff", "--diff-algorithm=histogram",
                       "--minimal", base_sha, head_sha)


def git_range_diff(repo_root: Path, old_sha: str, new_sha: str) -> bytes:
    """``git range-diff old...new`` -- how the commit series was rewritten
    between the two tips."""
    return _git_stdout(repo_root, "range-diff", "--no-color",
                       f"{old_sha}...{new_sha}")


def git_fetch_all(repo_root: Path) -> None:
    """``git fetch --all`` -- the same refresh fairy_fetch_git.sh runs
    on its schedule; a failed fetch raises rather than reading as
    success."""
    cp = subprocess.run(
        ["git", "-C", str(repo_root), "fetch", "--all", "--quiet"],
        check=False, text=True, capture_output=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"git fetch --all in {repo_root} failed: {cp.stderr.strip()}"
        )


def git_push_refspecs(
    repo_root: Path,
    remote_url: str,
    refspecs: Sequence[str],
    *,
    ssh_command: str | None = None,
    timeout_s: float = 60.0,
    force: bool = True,
    force_with_lease: str | None = None,
) -> None:
    """Push ``refspecs`` from ``repo_root`` to ``remote_url``.

    git negotiates a thin pack, so only objects the remote lacks are
    sent: the first push to a fresh mirror transfers full history, every
    later push transfers just the new commits. ``ssh_command`` overrides
    the transport (e.g. ``ssh -i <key> -o BatchMode=yes``) via
    ``GIT_SSH_COMMAND``. ``force`` (the default) passes ``--force``, as
    the mirror syncs need; pass False for a push that must refuse
    non-fast-forward updates. ``force_with_lease`` (a ``ref:sha``) makes
    the push succeed only while the remote ref still is that sha.
    """
    env = None
    if ssh_command is not None:
        env = {**os.environ, "GIT_SSH_COMMAND": ssh_command}
    cmd = ["git", "-C", str(repo_root), "push",
           *(["--force"] if force else []),
           *([f"--force-with-lease={force_with_lease}"]
             if force_with_lease else []),
           remote_url, *refspecs]
    cp = subprocess.run(cmd, env=env, capture_output=True, check=False, text=True, timeout=timeout_s)
    if cp.returncode != 0:
        raise RuntimeError(
            f"git push {' '.join(refspecs)} to {remote_url} from {repo_root} failed: "
            f"{cp.stderr.strip() or cp.stdout.strip()}"
        )


def list_head_tree_entries(repo_root: Path, revision: str) -> list[tuple[str, str]]:
    cp = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", revision],
        check=False,
        text=True,
        capture_output=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"failed to list files for {revision}")

    entries: list[tuple[str, str]] = []
    for line in cp.stdout.splitlines():
        if not line:
            continue
        try:
            meta, relpath = line.split("\t", 1)
        except ValueError:
            continue
        parts = meta.split()
        if len(parts) != 3:
            continue
        _mode, objtype, object_sha = parts
        if objtype != "blob":
            continue
        relpath = relpath.strip()
        object_sha = object_sha.strip()
        if relpath and object_sha:
            entries.append((relpath, object_sha))
    return entries
