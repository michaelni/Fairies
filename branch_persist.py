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

Persistence of reviewer-created git branches through fairy remotes.

Inside every review container each persistence-enabled repository's
checkout gets a remote named ``fairy``: a container-local bare repo
seeded with the published ``fairy/*`` branches, so the model
fetches and pushes there with plain git and no credentials. A branch
the verdict declares is packed into a thin git bundle -- its negatives
are commits the forge already has, so the bundle holds just the new
objects -- and the bundle rides base64-encoded inside the branch
record, which travels with the verdict into the filedb ticket. The
record is self-contained: publication and diffing rebuild the branch
from it wherever the ticket is, and it lives and dies with its ticket.
Records are applied to the forge under fairy's own ``fairy/``
namespace only when the review is approved and sent; until then the
branches exist nowhere another review, the client checkout, or the
forge could see them.

What does NOT belong here: the ``branches``/``pull_requests``
declaration vocabulary and its validation (``llm_review_api``),
pipeline wiring (the wrapper), mirror provisioning (``podman_repos``),
and forge REST access (gcli).
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from common import JsonObject
from git_util import FORGE_REMOTES, git_push_refspecs, git_rev_parse
from llm_review_api import (BRANCH_NAME_RE, BRANCH_TARGET_RE,
                            MAX_PERSIST_BRANCHES)
from podman_host import ContainerHandle, copy_into_container, run_on_remote_host
from podman_repos import RepoSpec

__all__ = [
    "FAIRY_BRANCH_PREFIX",
    "FAIRY_BRANCH_REMOTES",
    "MAX_BUNDLE_BYTES",
    "BranchTransferError",
    "CollectedBranch",
    "RemoteSeeds",
    "add_to_container_remote",
    "collect_declared_branches",
    "materialized_record",
    "publish_branch_record",
    "replace_invented_fairy_identities",
    "setup_container_remotes",
]

logger = logging.getLogger(__name__)

# How a record changes the published fairy branch: an ``ff`` push, a
# ``force`` push (the published tip the review saw is not an ancestor
# of the new one), or a ``delete``. Derived by collection against the
# seeded snapshot, never declared by the model.
BRANCH_MODES = ("ff", "force", "delete")

# The remote name each enabled checkout carries inside its container,
# and where its container-local bare repo lives. The prompt advertises
# the name; change it here and there together.
CONTAINER_REMOTE = "fairy"
_CONTAINER_QUARANTINE_DIR = "/quarantine"

# Every branch fairy publishes lives under this forge namespace; pushes
# outside it are impossible by construction, so a force-push can never
# touch a PR author's or protected branch.
FAIRY_BRANCH_PREFIX = "fairy/"

# Remotes whose fairy/* remote-tracking refs are the published fairy
# branches, in priority order: a deployment publishing into a dedicated
# fork keeps the fork as the checkout's "fairy" remote; one publishing
# into the reviewed repo itself sees them under its forge remotes.
FAIRY_BRANCH_REMOTES = ("fairy", *FORGE_REMOTES)

# Upper bound for one branch's bundle. Thin bundles hold only the
# objects the model added -- a few KB normally; a hostile container
# could synthesise unbounded history, so anything larger fails the
# collection, not the memory of every process the record passes.
MAX_BUNDLE_BYTES = 64 * 1024 * 1024

_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
_GIT_TIMEOUT_S = 600.0

# Reviewer models invent a git identity for fairy itself instead of
# leaving the configured one in effect (observed 2026-08-27..30 on six
# commits from several models: emails fairy@, forgejo_fairy@ or
# forgejo-fairy@ under made-up domains, overridden per command with
# git -c). Any other local part is somebody's real authorship and is
# never rewritten.
_INVENTED_FAIRY_EMAIL_LOCAL_RE = re.compile(rb"(forgejo[-_]?)?fairy",
                                            re.IGNORECASE)

_COMMIT_IDENT_RE = re.compile(
    rb"^(author|committer) .* <([^<>]*)> (\d+ [-+]\d{4})$")

# repo name -> {branch name -> sha}: a container's fairy remotes as
# seeded at provisioning, the baseline collection derives modes and
# bundle negatives from.
RemoteSeeds = dict[str, dict[str, str]]


class BranchTransferError(RuntimeError):
    """A branch could not be moved between container, record, or forge."""


@dataclass(frozen=True)
class CollectedBranch:
    """One declared branch packed out of a fairy remote: ``mode`` ff or
    force with the pushed tip's thin bundle in ``bundle`` (base64; empty
    when the tip introduces no new objects), or delete with ``sha``
    naming the published tip to remove and no bundle. ``old_sha`` is the
    seeded ``fairy/<branch>`` tip the mode was derived against (None for
    a new branch); ``objects_repo`` the local checkout whose objects
    satisfy the bundle's prerequisites.

    ``record()`` is the JSON-safe shape that crosses process boundaries
    (wrapper stdout, filedb tickets).
    """

    branch: str
    mode: str
    pr: JsonObject | None
    repo: str
    sha: str
    old_sha: str | None
    bundle: str
    objects_repo: str
    diff_base_sha: str | None

    def record(self) -> JsonObject:
        return {"branch": self.branch, "mode": self.mode, "pr": self.pr,
                "repo": self.repo, "sha": self.sha, "old_sha": self.old_sha,
                "bundle": self.bundle, "objects_repo": self.objects_repo,
                "diff_base_sha": self.diff_base_sha}


def _git(repo: Path, *args: str, check: bool = True,
         timeout_s: float = _GIT_TIMEOUT_S,
         input_bytes: bytes | None = None,
         binary: bool = False) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(repo), *args]
    logger.debug("branch record git: %s", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=not binary,
                        input=input_bytes, check=False, timeout=timeout_s)
    if check and cp.returncode != 0:
        stderr = cp.stderr.decode(errors="replace") if binary else cp.stderr
        stdout = cp.stdout.decode(errors="replace") if binary else cp.stdout
        raise BranchTransferError(
            f"git {' '.join(args)} in {repo} failed (rc={cp.returncode}): "
            f"{stderr.strip() or stdout.strip()}")
    return cp


def _container_git(handle: ContainerHandle, repo_path: str, *args: str,
                   timeout_s: float = _GIT_TIMEOUT_S,
                   max_output_bytes: int | None = None):
    return run_on_remote_host(
        handle.host, "podman", "exec", handle.container_id,
        "git", "-C", repo_path, *args, timeout_s=timeout_s,
        max_output_bytes=max_output_bytes)


def _fake_remote_path(repo_name: str) -> str:
    return f"{_CONTAINER_QUARANTINE_DIR}/{repo_name}.git"


def _parse_ref_listing(listing: str, where: str) -> dict[str, str]:
    """{branch: sha} from a ``for-each-ref`` listing whose content came
    from an untrusted container: only names in the safe charset and
    well-formed SHAs survive."""
    refs: dict[str, str] = {}
    for line in listing.splitlines():
        name, _, sha = line.partition(" ")
        if BRANCH_NAME_RE.fullmatch(name) and _SHA_RE.fullmatch(sha):
            refs[name] = sha
        elif line.strip():
            logger.warning("ignoring ref with unsafe name/sha in %s: %r",
                           where, line[:120])
    return refs


def _list_fake_refs(handle: ContainerHandle, repo_name: str) -> dict[str, str]:
    """The in-container fairy remote's branches as {name: sha}."""
    cp = _container_git(handle, _fake_remote_path(repo_name),
                        "for-each-ref", "--format=%(refname:short) %(objectname)",
                        "refs/heads", max_output_bytes=MAX_BUNDLE_BYTES)
    if cp.returncode != 0:
        raise BranchTransferError(
            f"listing the fairy remote of {repo_name!r} in container "
            f"{handle.container_id[:12]} failed: "
            f"{cp.stderr.decode(errors='replace').strip()}")
    return _parse_ref_listing(cp.stdout.decode(errors="replace"), repo_name)


def setup_container_remotes(
    handle: ContainerHandle,
    repo_specs: Sequence[RepoSpec],
    enabled: Sequence[str],
) -> RemoteSeeds:
    """Create each enabled repo's fairy remote in the container: a bare
    repo whose alternates read the checkout's objects, seeded with the
    published ``fairy/*`` branches (as the checkout's remote-tracking
    refs of ``FAIRY_BRANCH_REMOTES`` know them), wired up as the remote
    ``CONTAINER_REMOTE``. Returns the seeded {repo: {branch: sha}}
    snapshot collection derives modes and bundle negatives from."""
    seeds: RemoteSeeds = {}
    for spec in repo_specs:
        if spec.name not in enabled:
            continue
        fake = _fake_remote_path(spec.name)
        cp = run_on_remote_host(handle.host, "podman", "exec",
                                handle.container_id,
                                "git", "init", "--bare", "--quiet", fake)
        if cp.returncode == 0:
            # the bare repo's alternates read the checkout's objects, so
            # it stores only what the model pushes on top
            cp = run_on_remote_host(
                handle.host, "podman", "exec", handle.container_id, "sh", "-c",
                f"echo {spec.container_path}/.git/objects"
                f" > {fake}/objects/info/alternates")
        if cp.returncode != 0:
            raise BranchTransferError(
                f"creating the fairy remote for {spec.name!r} in container "
                f"{handle.container_id[:12]} failed: "
                f"{cp.stderr.decode(errors='replace').strip()}")
        # a pattern refspec that matches nothing pushes nothing; the
        # priority remote pushes last, so it wins name collisions
        for remote in reversed(FAIRY_BRANCH_REMOTES):
            _container_git(
                handle, spec.container_path, "push", "--quiet", fake,
                f"+refs/remotes/{remote}/{FAIRY_BRANCH_PREFIX}*:refs/heads/*")
        if _container_git(handle, spec.container_path, "remote", "add",
                          CONTAINER_REMOTE, fake).returncode != 0:
            _container_git(handle, spec.container_path, "remote", "set-url",
                           CONTAINER_REMOTE, fake)
        seeds[spec.name] = _list_fake_refs(handle, spec.name)
        logger.info("fairy remote for %r in container %s seeded with %d "
                    "branch(es)", spec.name, handle.container_id[:12],
                    len(seeds[spec.name]))
    return seeds


def add_to_container_remote(
    handle: ContainerHandle,
    records: Sequence[JsonObject],
    repo_container_paths: dict[str, str],
) -> None:
    """Apply branch records to the container's fairy remotes -- the
    drafts' branches onto the combiner's remotes, after its baseline
    seeds were taken, so the combiner verifies them like its own work
    and re-declares what should survive. A later record overwrites an
    earlier same-named branch."""
    for record in records:
        repo, branch = record.get("repo"), record.get("branch")
        if repo not in repo_container_paths:
            continue
        fake = _fake_remote_path(repo)
        if record.get("mode") == "delete":
            _container_git(handle, fake, "update-ref", "-d",
                           f"refs/heads/{branch}")
            continue
        if not record.get("bundle"):
            # no new objects: the tip is reachable from the checkout the
            # remote's alternates read
            cp = _container_git(handle, fake, "update-ref",
                                f"refs/heads/{branch}", str(record.get("sha")))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                bundle_path = Path(tmp) / f"{repo}-{branch}.bundle"
                bundle_path.write_bytes(
                    base64.b64decode(str(record.get("bundle"))))
                copy_into_container(handle, bundle_path,
                                    _CONTAINER_QUARANTINE_DIR)
            cp = _container_git(
                handle, fake, "fetch", "--no-tags",
                f"{_CONTAINER_QUARANTINE_DIR}/{bundle_path.name}",
                f"+refs/heads/{branch}:refs/heads/{branch}")
        if cp.returncode != 0:
            raise BranchTransferError(
                f"adding branch {branch!r} of {repo!r} to the fairy remote "
                f"failed: {cp.stderr.decode(errors='replace').strip()}")
        logger.info("added branch %r of %r to the fairy remote in "
                    "container %s", branch, repo, handle.container_id[:12])


def _merge_base(handle: ContainerHandle, repo: str, sha: str,
                other: str | None) -> str | None:
    """The merge base of ``sha`` and ``other`` in the repo's fairy
    remote; None when there is none (or no ``other``)."""
    if other is None:
        return None
    cp = _container_git(handle, _fake_remote_path(repo), "merge-base",
                        sha, other)
    merge_base = cp.stdout.decode(errors="replace").strip()
    if cp.returncode != 0 or not _SHA_RE.fullmatch(merge_base):
        return None
    return merge_base


def _forge_branch_tips(handle: ContainerHandle,
                       container_path: str) -> dict[str, str]:
    """{remote-qualified branch name -> sha} for every branch the
    checkout's forge remotes track, PR head refs excluded -- the commits
    the forge is known to have, where a PR target resolves, and the
    negatives that keep a bundle to the model's own commits. The
    container remote's tracking refs follow what the model pushed and
    fetched there, so they are not consulted."""
    cp = _container_git(
        handle, container_path, "for-each-ref",
        "--format=%(refname:short) %(objectname)",
        *(f"refs/remotes/{remote}" for remote in FORGE_REMOTES),
        max_output_bytes=MAX_BUNDLE_BYTES)
    if cp.returncode != 0:
        raise BranchTransferError(
            f"listing the forge branches of {container_path!r} failed: "
            f"{cp.stderr.decode(errors='replace').strip()}")
    tips: dict[str, str] = {}
    for line in cp.stdout.decode(errors="replace").splitlines():
        name, _, sha = line.partition(" ")
        _, _, branch_part = name.partition("/")
        if branch_part.startswith("pr/") or not branch_part:
            continue
        if BRANCH_TARGET_RE.fullmatch(name) and _SHA_RE.fullmatch(sha):
            tips[name] = sha
    return tips


def bundle_prerequisites(bundle: bytes) -> list[str]:
    """The prerequisite commits a bundle's header names (its ``-<sha>``
    lines): with every forge branch tip among the create's negatives,
    a single prerequisite is the branch's fork point off the forge --
    the natural preview base of a push that names no PR target."""
    prerequisites = []
    for line in bundle.split(b"\n"):
        if not line:
            break
        if line.startswith(b"-"):
            sha = line[1:].split(b" ", 1)[0].decode("ascii", "replace")
            if _SHA_RE.fullmatch(sha):
                prerequisites.append(sha)
    return prerequisites


def _bundle_from_container(handle: ContainerHandle, repo: str, branch: str,
                           negatives: Sequence[str]) -> str:
    """The branch's thin bundle out of the container's fairy remote,
    base64-encoded; empty when the tip adds no objects over the
    negatives (git 2.43, observed: "fatal: Refusing to create empty
    bundle."), whose ref the record's SHA alone can publish."""
    cp = _container_git(
        handle, _fake_remote_path(repo), "bundle", "create", "-",
        f"refs/heads/{branch}", *(("--not", *negatives) if negatives else ()),
        max_output_bytes=MAX_BUNDLE_BYTES)
    if cp.returncode != 0:
        stderr = cp.stderr.decode(errors="replace").strip()
        if "empty bundle" in stderr:
            return ""
        raise BranchTransferError(
            f"bundling {branch!r} of {repo!r} failed: {stderr}")
    return base64.b64encode(cp.stdout).decode("ascii")


def collect_declared_branches(
    seeded_handles: Sequence[tuple[ContainerHandle, RemoteSeeds]],
    declared: Sequence[JsonObject],
    pull_requests: Sequence[JsonObject],
    *,
    repo_specs: Sequence[RepoSpec],
    base_shas: dict[str, Sequence[str]],
) -> list[CollectedBranch]:
    """Pack the verdict's declared branches out of the containers' fairy
    remotes into self-contained records.

    ``declared`` and ``pull_requests`` are the role's sanitized lists
    (``llm_review_api``); a pull request implies its branch. Every
    branch tip the checkout's remotes track joins the bundle negatives,
    so a bundle holds the model's own commits alone; a PR record's
    preview base is the merge base with its target branch, a plain
    push's the bundle's single prerequisite -- its fork point off the
    forge. ``base_shas`` maps each repo to further forge-known commits
    (e.g. a PR head under review). Model mistakes -- a declared branch no fairy
    remote holds, a deletion of an unpublished branch, more than
    ``MAX_PERSIST_BRANCHES`` per repo -- are dropped with a warning;
    infrastructure failures raise.
    """
    specs = {spec.name: spec for spec in repo_specs}
    requests_left = {(r["repo"], r["branch"]): r for r in pull_requests}
    wanted = [(d["repo"], d["branch"], d["action"]) for d in declared]
    wanted += [(repo, branch, "push") for (repo, branch) in requests_left
               if not any(w[:2] == (repo, branch) for w in wanted)]
    listings: dict[tuple[int, str], dict[str, str]] = {}
    tip_cache: dict[tuple[int, str], dict[str, str]] = {}
    counts: dict[str, int] = {}
    records: list[CollectedBranch] = []
    for repo, branch, action in wanted:
        spec = specs.get(repo)
        if spec is None:
            logger.warning("declared branch %r names no persistence-enabled "
                           "repo %r; dropped", branch, repo)
            continue
        pr = requests_left.pop((repo, branch), None)
        if counts.get(repo, 0) >= MAX_PERSIST_BRANCHES:
            logger.warning("dropped %r beyond the %d-branch cap of %r",
                           branch, MAX_PERSIST_BRANCHES, repo)
            continue
        if action == "delete":
            old_sha = next((s[repo].get(branch) for _, s in seeded_handles
                            if repo in s), None)
            if pr is not None:
                logger.warning("dropped pull request for declared deletion "
                               "of %r of %r", branch, repo)
            if old_sha is None:
                logger.warning("declared deletion of %r of %r, which is not "
                               "published; dropped", branch, repo)
                continue
            records.append(CollectedBranch(
                branch=branch, mode="delete", pr=None, repo=repo, sha=old_sha,
                old_sha=old_sha, bundle="", objects_repo=str(spec.repo_root),
                diff_base_sha=None))
            logger.info("collected deletion of %r from %r", branch, repo)
        else:
            found = None
            for handle, handle_seeds in seeded_handles:
                if repo not in handle_seeds:
                    continue
                refs = listings.setdefault((id(handle), repo),
                                           _list_fake_refs(handle, repo))
                if branch in refs:
                    # this handle's own seeds: another machine's mirror
                    # may have differing fairy/* tips whose objects this
                    # container does not hold
                    found = (handle, handle_seeds[repo], refs[branch])
                    break
            if found is None:
                logger.warning("declared branch %r is on no fairy remote of "
                               "%r; dropped", branch, repo)
                continue
            handle, seeds, sha = found
            old_sha = seeds.get(branch)
            if sha == old_sha:
                logger.info("declared branch %r of %r matches the published "
                            "tip; nothing to publish", branch, repo)
                continue
            mode = "ff"
            if old_sha is not None:
                ancestor_rc = _container_git(
                    handle, _fake_remote_path(repo), "merge-base",
                    "--is-ancestor", old_sha, sha).returncode
                if ancestor_rc not in (0, 1):
                    raise BranchTransferError(
                        f"deriving the push mode of {branch!r} of {repo!r} "
                        f"failed (rc={ancestor_rc})")
                mode = "ff" if ancestor_rc == 0 else "force"
            tips = tip_cache.setdefault(
                (id(handle), repo),
                _forge_branch_tips(handle, spec.container_path))
            target = (pr or {}).get("target")
            target_tip = next(
                (tips[f"{remote}/{target}"] for remote in FORGE_REMOTES
                 if f"{remote}/{target}" in tips), None) if target else None
            negatives = sorted({*seeds.values(),
                                *(base_shas.get(repo) or ()),
                                *tips.values()} - {sha})
            bundle = _bundle_from_container(handle, repo, branch, negatives)
            if pr is not None:
                # a PR's one true preview base: the merge base with its
                # target branch -- no base at all beats a wrong one
                diff_base = _merge_base(handle, repo, sha, target_tip)
            else:
                prerequisites = bundle_prerequisites(
                    base64.b64decode(bundle))
                diff_base = prerequisites[0] \
                    if len(prerequisites) == 1 else None
            records.append(CollectedBranch(
                branch=branch, mode=mode, pr=pr, repo=repo, sha=sha,
                old_sha=old_sha, bundle=bundle,
                objects_repo=str(spec.repo_root), diff_base_sha=diff_base))
            logger.info("collected %s of %r from %r sha=%s bundle=%dB "
                        "diff_base=%s", mode, branch, repo, sha[:12],
                        len(bundle) * 3 // 4, (diff_base or "-")[:12])
        counts[repo] = counts.get(repo, 0) + 1
    return records


def _checked_record(record: JsonObject) -> tuple[str, str]:
    """Boundary check for a record read back from a filedb ticket, which
    is hand-editable: safe branch name, known mode, well-formed SHAs, a
    bounded base64 bundle. Returns (branch, sha)."""
    branch, mode = record.get("branch"), record.get("mode")
    sha, old_sha = record.get("sha"), record.get("old_sha")
    bundle = record.get("bundle")
    if not isinstance(branch, str) or not BRANCH_NAME_RE.fullmatch(branch) \
            or mode not in BRANCH_MODES \
            or not isinstance(sha, str) or not _SHA_RE.fullmatch(sha) \
            or not (old_sha is None or (isinstance(old_sha, str)
                                        and _SHA_RE.fullmatch(old_sha))) \
            or (mode == "force" and old_sha is None) \
            or not isinstance(bundle, str) \
            or len(bundle) > MAX_BUNDLE_BYTES * 4 // 3 + 4:
        raise BranchTransferError(
            "invalid branch record: "
            f"{ {k: v for k, v in record.items() if k != 'bundle'} }")
    return branch, sha


@contextmanager
def materialized_record(record: JsonObject,
                        *, extra_objects: Path | None = None) -> Iterator[Path]:
    """A scratch bare repo holding a push record's tip under
    ``refs/heads/<branch>``: the record's bundle fetched with its
    prerequisites read from the ``objects_repo`` checkout (and
    ``extra_objects``, e.g. the operator's patch repo, for diffing
    against refs only that repo has). The repo is removed on exit.
    Raises ``BranchTransferError`` on a malformed record, a missing
    prerequisite, or a bundled tip that is not the recorded ``sha``."""
    branch, sha = _checked_record(record)
    with tempfile.TemporaryDirectory(prefix="fairy-branch-") as tmp:
        scratch = Path(tmp) / "record.git"
        _git(Path(tmp), "init", "--bare", "--quiet", str(scratch))
        alternates = []
        for objects_repo in (record.get("objects_repo"), extra_objects):
            if not objects_repo:
                continue
            cp = _git(Path(objects_repo), "rev-parse", "--path-format=absolute",
                      "--git-path", "objects", check=False)
            if cp.returncode == 0:
                alternates.append(cp.stdout.strip())
        if alternates:
            (scratch / "objects" / "info" / "alternates").write_text(
                "\n".join(alternates) + "\n", encoding="utf-8")
        if record.get("bundle"):
            bundle_path = Path(tmp) / "record.bundle"
            try:
                bundle_path.write_bytes(
                    base64.b64decode(str(record["bundle"]), validate=True))
            except (binascii.Error, ValueError) as exc:
                raise BranchTransferError(
                    f"the bundle of {branch!r} is not base64: {exc}") from exc
            _git(scratch, "fetch", "--no-tags", str(bundle_path),
                 f"+refs/heads/{branch}:refs/heads/{branch}")
        else:
            _git(scratch, "update-ref", f"refs/heads/{branch}", sha)
        try:
            current = git_rev_parse(scratch, f"refs/heads/{branch}")
        except RuntimeError as exc:
            raise BranchTransferError(
                f"materializing {branch!r} yielded no tip: {exc}") from exc
        if current != sha:
            raise BranchTransferError(
                f"the bundle of {branch!r} holds {current[:12]}, the record "
                f"says {sha[:12]}")
        yield scratch


def replace_invented_fairy_identities(
        record: JsonObject, commit_author: tuple[str, str]) -> JsonObject:
    """The push record with every bundled commit whose author or
    committer email has a local part matching
    ``_INVENTED_FAIRY_EMAIL_LOCAL_RE`` rewritten to ``commit_author``
    ((name, email)), rebundled under the original prerequisites; the
    record itself when its commits are clean, a delete, or an empty
    bundle. Raises ``BranchTransferError`` when the record cannot be
    materialized or rewritten."""
    if record.get("mode") == "delete" or not record.get("bundle"):
        return record
    author_name, author_email = commit_author
    configured = f"{author_name} <{author_email}>".encode()
    prerequisites = bundle_prerequisites(
        base64.b64decode(str(record["bundle"])))
    branch, tip_sha = str(record["branch"]), str(record["sha"])
    with materialized_record(record) as scratch:
        # --topo-order: the parent remap needs every parent rewritten
        # before its children; the default date order breaks that on a
        # merge whose inner parent carries a newer commit date
        # (observed with git 2.43)
        bundled_commits = _git(
            scratch, "rev-list", "--reverse", "--topo-order", tip_sha,
            *(("--not", *prerequisites) if prerequisites else ()),
        ).stdout.split()
        rewritten: dict[bytes, bytes] = {}
        for commit in bundled_commits:
            raw = _git(scratch, "cat-file", "commit", commit,
                       binary=True).stdout
            header, separator, message = raw.partition(b"\n\n")
            lines = header.split(b"\n")
            changed = False
            for i, line in enumerate(lines):
                ident = _COMMIT_IDENT_RE.match(line)
                if ident and _INVENTED_FAIRY_EMAIL_LOCAL_RE.fullmatch(
                        ident.group(2).split(b"@", 1)[0]):
                    lines[i] = b" ".join(
                        (ident.group(1), configured, ident.group(3)))
                    changed = True
                elif line.startswith(b"parent ") and line[7:] in rewritten:
                    lines[i] = b"parent " + rewritten[line[7:]]
                    changed = True
            if changed:
                rewritten[commit.encode("ascii")] = _git(
                    scratch, "hash-object", "-t", "commit", "-w", "--stdin",
                    binary=True,
                    input_bytes=b"\n".join(lines) + separator + message,
                ).stdout.strip()
        new_tip = rewritten.get(tip_sha.encode("ascii"))
        if new_tip is None:
            return record
        new_sha = new_tip.decode("ascii")
        _git(scratch, "update-ref", f"refs/heads/{branch}", new_sha)
        bundle_path = scratch.parent / "rewritten.bundle"
        _git(scratch, "bundle", "create", str(bundle_path),
             f"refs/heads/{branch}",
             *(("--not", *prerequisites) if prerequisites else ()))
        bundle = base64.b64encode(bundle_path.read_bytes()).decode("ascii")
    logger.info("invented fairy identities: rewrote %d of %d commit(s) "
                "of %r: %s -> %s", len(rewritten), len(bundled_commits),
                branch, tip_sha[:12], new_sha[:12])
    return {**record, "sha": new_sha, "bundle": bundle}


def publish_branch_record(
    record: JsonObject,
    *,
    remote_url: str,
    timeout_s: float = 300.0,
) -> str:
    """Apply one branch record to ``remote_url``: push the recorded tip
    as ``fairy/<branch>``, or delete the published branch. Returns the
    forge branch name.

    An ``ff`` record pushes plainly, so it fails once ``fairy/<branch>``
    moved since collection -- two reviews can both look fast-forward,
    whichever is approved first wins and the second is no fast-forward
    any more. ``force`` and ``delete`` push under ``--force-with-lease``
    of the recorded old tip, so they only ever overwrite the state the
    review saw. A delete whose branch is already gone passes, keeping
    retried sends idempotent.
    """
    branch, sha = _checked_record(record)
    forge_branch = f"{FAIRY_BRANCH_PREFIX}{branch}"
    forge_ref = f"refs/heads/{forge_branch}"
    if record["mode"] == "delete":
        with tempfile.TemporaryDirectory(prefix="fairy-branch-") as tmp:
            scratch = Path(tmp) / "record.git"
            _git(Path(tmp), "init", "--bare", "--quiet", str(scratch))
            logger.info("deleting %s (%s) from %s", forge_branch, sha[:12],
                        remote_url)
            try:
                git_push_refspecs(scratch, remote_url, [f":{forge_ref}"],
                                  force=False,
                                  force_with_lease=f"{forge_ref}:{sha}",
                                  timeout_s=timeout_s)
            except RuntimeError as exc:
                listed = _git(scratch, "ls-remote", remote_url, forge_ref,
                              timeout_s=timeout_s).stdout.split()
                if listed:
                    raise BranchTransferError(
                        f"{forge_branch} moved on {remote_url}: ticket would "
                        f"delete {sha[:12]}, forge has {listed[0][:12]}") from exc
                logger.info("%s is already gone from %s", forge_branch,
                            remote_url)
        return forge_branch
    with materialized_record(record) as scratch:
        logger.info("pushing %s (%s) %s -> %s %s", branch, record["mode"],
                    sha[:12], remote_url, forge_branch)
        lease = f"{forge_ref}:{record['old_sha']}" \
            if record["mode"] == "force" else None
        try:
            git_push_refspecs(scratch, remote_url, [f"{sha}:{forge_ref}"],
                              force=False, force_with_lease=lease,
                              timeout_s=timeout_s)
        except RuntimeError as exc:
            raise BranchTransferError(str(exc)) from exc
    return forge_branch
