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

Pure-text parsing of ``git format-patch`` output.

What belongs here: splitting a patch into per-file diff blocks and
extracting changed paths, commit SHAs, and submodule (gitlink) changes
from them. No subprocess calls, no SDKs, no prompt text -- every
consumer (prompt building, source-bundle assembly) supplies the patch
string and interprets the results itself.

This is a leaf module: it has no in-repo dependencies.
"""

from __future__ import annotations

import re

__all__ = [
    "extract_changed_paths_from_patch",
    "extract_commit_shas_from_patch",
    "extract_submodule_changes_from_patch",
    "extract_submodule_paths_from_patch",
]


_DIFF_GIT_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)


def _split_patch_into_diff_blocks(patch: str) -> list[str]:
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    return [p for p in parts if p.startswith("diff --git ")]


def _diff_block_own_body(block: str) -> str:
    """Return only the portion of ``block`` that belongs to its own diff.

    A patch produced by ``git format-patch`` concatenates multiple commits.
    After the last ``diff --git`` of commit N, the stream includes the
    ``-- \\n<version>`` separator, then commit N+1's ``From <sha>`` header
    and its per-commit file-stats (which may contain lines like
    ``create mode 160000 path/to/submodule``). Splitting the patch on
    ``^diff --git `` therefore returns a block that ends somewhere inside
    commit N+1's header, so a naive ``"mode 160000" in block`` check
    would misattribute commit N+1's summary line to commit N's last file.

    Truncate at either the patch trailer (``\\n-- \\n``) or the next
    commit header (``^From <sha>``) so callers only scan lines that
    actually describe this file's diff.
    """
    cut = len(block)
    m = re.search(r"(?m)^-- $", block)
    if m:
        cut = min(cut, m.start())
    m = re.search(r"(?m)^From [0-9a-fA-F]{7,64} ", block)
    if m:
        cut = min(cut, m.start())
    return block[:cut]


_GITLINK_MODE_RE = re.compile(
    r"(?m)^(?:new file mode|deleted file mode|old mode|new mode) 160000\b"
)
_GITLINK_INDEX_RE = re.compile(r"(?m)^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+ 160000\b")
_GITLINK_SUBPROJECT_RE = re.compile(r"(?m)^[+\- ]Subproject commit [0-9a-fA-F]+")


def _diff_block_is_submodule(block: str) -> bool:
    """Detect diff blocks that describe a gitlink (submodule) change.

    Git represents submodules with mode 160000 in three distinguishable
    ways depending on what kind of change the diff carries:

    - new submodule:  ``new file mode 160000``
    - deleted submodule:  ``deleted file mode 160000``
    - bumped commit:  ``index <old>..<new> 160000`` plus ``-Subproject
      commit <old>`` / ``+Subproject commit <new>`` hunk lines

    Any of those signals identifies the block as a gitlink. None of
    them have ``git show``-able file content, so the wrapper must skip
    them before ``load_source_bundle_texts`` tries to resolve them.
    """
    body = _diff_block_own_body(block)
    return bool(
        _GITLINK_MODE_RE.search(body)
        or _GITLINK_INDEX_RE.search(body)
        or _GITLINK_SUBPROJECT_RE.search(body)
    )


_GITLINK_SUBPROJECT_LINE_RE = re.compile(
    r"(?m)^([+\- ])Subproject commit ([0-9a-fA-F]+)\s*$"
)


def _diff_block_submodule_action(body: str) -> str:
    """Classify a submodule diff block as ``added``/``removed``/``updated``."""
    if "new file mode 160000" in body:
        return "added"
    if "deleted file mode 160000" in body:
        return "removed"
    return "updated"


def _diff_block_subproject_commits(body: str) -> tuple[str | None, str | None]:
    """Return ``(old_commit, new_commit)`` parsed from a submodule diff hunk.

    Either side may be ``None`` if the patch only carries the other side
    (e.g. a brand-new submodule has no ``-Subproject commit`` line, a
    deletion has no ``+`` line). Hex SHAs are returned verbatim.
    """
    old_commit: str | None = None
    new_commit: str | None = None
    for sign, sha in _GITLINK_SUBPROJECT_LINE_RE.findall(body):
        if sign == "+":
            new_commit = sha
        elif sign == "-":
            old_commit = sha
    return old_commit, new_commit


def extract_submodule_changes_from_patch(patch: str) -> list[dict[str, str | None]]:
    """Return structured per-submodule change records from a patch.

    Each record is ``{"path": str, "action": "added"|"removed"|"updated",
    "old_commit": str|None, "new_commit": str|None}``. Records are
    deduplicated by path and ordered by first appearance in the patch.
    Surfaced to the LLM reviewer as explicit metadata because submodule
    additions pull external code into the tree and warrant explicit
    human attention; the raw patch alone exposes the change only as
    terse ``mode 160000`` / ``Subproject commit <sha>`` markers that
    are easy to miss inside a large diff.
    """
    seen: set[str] = set()
    result: list[dict[str, str | None]] = []
    for block in _split_patch_into_diff_blocks(patch):
        if not _diff_block_is_submodule(block):
            continue
        header = _DIFF_GIT_HEADER_RE.search(block)
        if header is None:
            continue
        # Use the b/-side path as the canonical one for added/updated
        # entries; for a removed submodule the b/-side equals the a/-side
        # so the choice does not matter.
        path = header.group(2).strip()
        if path in seen:
            continue
        seen.add(path)
        body = _diff_block_own_body(block)
        old_commit, new_commit = _diff_block_subproject_commits(body)
        result.append(
            {
                "path": path,
                "action": _diff_block_submodule_action(body),
                "old_commit": old_commit,
                "new_commit": new_commit,
            }
        )
    return result


def extract_submodule_paths_from_patch(patch: str) -> set[str]:
    result: set[str] = set()
    for block in _split_patch_into_diff_blocks(patch):
        if not _diff_block_is_submodule(block):
            continue
        header = _DIFF_GIT_HEADER_RE.search(block)
        if header is None:
            continue
        result.add(header.group(1).strip())
        result.add(header.group(2).strip())
    return result


def extract_changed_paths_from_patch(patch: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    submodule_paths = extract_submodule_paths_from_patch(patch)

    patterns = [
        re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE),
        _DIFF_GIT_HEADER_RE,
    ]

    for match in patterns[0].finditer(patch):
        path = match.group(1).strip()
        if path == "/dev/null" or path in submodule_paths:
            continue
        if path not in seen:
            seen.add(path)
            paths.append(path)

    for match in patterns[1].finditer(patch):
        old_path, new_path = match.group(1).strip(), match.group(2).strip()
        candidate = new_path if new_path != "/dev/null" else old_path
        if candidate == "/dev/null" or candidate in submodule_paths:
            continue
        if candidate not in seen:
            seen.add(candidate)
            paths.append(candidate)

    return paths


def extract_commit_shas_from_patch(patch: str) -> list[str]:
    commit_shas: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?m)^From ([0-9a-fA-F]{7,64})\b", patch):
        commit_sha = match.group(1)
        if commit_sha not in seen:
            seen.add(commit_sha)
            commit_shas.append(commit_sha)
    return commit_shas
