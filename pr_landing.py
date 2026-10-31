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

Where a pull request's patches landed: its commits matched by content
against the commits of the forge's branches, so a pull request closed by
hand after its patches were pushed is told from one that was rejected.

Each pull request commit is scored against candidate landed commits by
four tests:

  P  equal git patch-id (the diff hash, stable across a rebase)   2 points
  S  equal subject (first line)                                   1 point
  A  equal author date (preserved by a rebase/cherry-pick)        1 point
  C  landed commit date no earlier than the pull request's        2 points
     (within a day, to absorb timezone slop)

Match tiers, strongest first, are an exact SHA (the commit itself is on
the branch) then total score 6, 5 and 4. The four tests form a signature,
e.g. "PSAC" (all four), "P.AC" (subject differs) or "SHA" for an exact
SHA; a "." marks a failed test.

A landed commit matches only one pull request commit: once a stronger
tier claims it, it is no longer offered to weaker ones. When several
pull request commits claim it within a tier none can be singled out and
those matches are uncertain.

What belongs here: the scoring, the tier resolution and the index of
the forge's branches with their cached patch-ids.
What does NOT belong: reading pull requests from the forge or the filedb
snapshots (agent) and reporting (tools/match_prs_to_master).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Container, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import filedb
import git_util

__all__ = ["Commit", "Index", "resolve", "forge_commits", "log_commits"]

logger = logging.getLogger(__name__)

SHA_SCORE = 7
TIERS = (SHA_SCORE, 6, 5, 4)
COMMIT_DATE_TOLERANCE_S = 24 * 3600


class Commit(NamedTuple):
    subject: str
    author_date: int
    commit_date: int


class Index:
    """The landed commits a pull request commit is matched against.

    commits:   {sha: Commit} of the landed commits, in git log order
    patch_ids: {sha: patch-id}, a commit without a diff absent
    reachable: every sha on the branches, for the exact-SHA test;
               the keys of ``commits`` when omitted
    """

    def __init__(self, commits: dict[str, Commit], patch_ids: dict[str, str],
                 reachable: Container[str] | None = None) -> None:
        self.commits = commits
        self.reachable = commits if reachable is None else reachable
        self.by_patch_id: dict[str, list[str]] = defaultdict(list)
        self.by_subject: dict[str, list[str]] = defaultdict(list)
        for sha, commit in commits.items():
            if sha in patch_ids:
                self.by_patch_id[patch_ids[sha]].append(sha)
            self.by_subject[commit.subject].append(sha)

    def candidates(self, sha: str, commit: Commit,
                   patch_id: str | None) -> dict[str, tuple[int, str]]:
        """The landed commits ``sha`` could be, as {landed_sha: (score,
        signature)}, scoring at least the weakest tier."""
        if sha in self.reachable:
            return {sha: (SHA_SCORE, "SHA")}
        pool = set(self.by_patch_id.get(patch_id, ())) \
            | set(self.by_subject.get(commit.subject, ()))
        out = {}
        for landed_sha in pool:
            landed = self.commits[landed_sha]
            tests = (landed_sha in self.by_patch_id.get(patch_id, ()),
                     commit.subject == landed.subject,
                     commit.author_date == landed.author_date,
                     landed.commit_date >= commit.commit_date - COMMIT_DATE_TOLERANCE_S)
            score = 2 * tests[0] + tests[1] + tests[2] + 2 * tests[3]
            if score >= TIERS[-1]:
                out[landed_sha] = (score, "".join(
                    c if t else "." for c, t in zip("PSAC", tests)))
        return out


def resolve(entries: Sequence[tuple[str, dict[str, tuple[int, str]]]]
            ) -> list[tuple[dict[str, str], bool]]:
    """Resolve the candidates of several pull request commits in score
    tiers: a landed commit taken by a stronger match is no longer
    offered to a weaker one; one claimed by several different commits
    within a tier is consumed but their matches are uncertain.

    entries: (sha, candidates) per pull request commit, the candidates
             as Index.candidates gives them
    returns: per entry its matches {landed_sha: signature}, {} when
             none, and whether they are certain
    """
    matched: dict[int, tuple[dict[str, str], bool]] = {}
    consumed: set[str] = set()
    for tier in TIERS:
        won = {}
        claimants: dict[str, set[str]] = defaultdict(set)
        for i, (sha, candidates) in enumerate(entries):
            if i in matched:
                continue
            available = {landed_sha: signature
                         for landed_sha, (score, signature) in candidates.items()
                         if score == tier and landed_sha not in consumed}
            if available:
                won[i] = available
                for landed_sha in available:
                    claimants[landed_sha].add(sha)
        for i, available in won.items():
            matched[i] = (available, all(len(claimants[landed_sha]) == 1
                                         for landed_sha in available))
        consumed.update(landed_sha for available in won.values() for landed_sha in available)
    return [matched.get(i, ({}, True)) for i in range(len(entries))]


def log_commits(repo_root: Path, *args: str,
                stdin: Sequence[str] | None = None) -> dict[str, Commit]:
    """{sha: Commit} of ``git log <args>``, in log order; ``stdin`` names
    commits to read instead, one that the repository lacks absent."""
    if stdin is not None and not stdin:
        return {}
    out = git_util.git_log(
        repo_root, "--format=%H%x1f%at%x1f%ct%x1f%s",
        *(("--no-walk", "--ignore-missing", "--stdin") if stdin is not None else ()),
        *args, stdin="".join(f"{sha}\n" for sha in stdin or ()))
    commits = {}
    for line in out.splitlines():
        sha, author_date, commit_date, subject = line.split("\x1f", 3)
        commits[sha] = Commit(subject, int(author_date), int(commit_date))
    return commits


LANDED_DOC = "landed-patch-ids.json"


def forge_commits(db: filedb.Db, repo_root: Path, since: datetime
                  ) -> tuple[dict[str, Commit], dict[str, str]]:
    """The commits of the forge's branches, pull request refs excluded.

    db:        the db whose root caches the sha-to-patch-id map
    repo_root: a clone with the forge remote (see git_util.FORGE_REMOTES)
    since:     only non-merge commits from this time on
    returns:   {sha: Commit} and {sha: patch-id}
    """
    remote = git_util.git_forge_remote(repo_root)
    commits = log_commits(repo_root, "--no-merges", f"--since={since.isoformat()}",
                          f"--exclude={remote}/pr/*", f"--remotes={remote}")
    known = db.read(db.root / LANDED_DOC) or {}
    ids = {sha: known[sha] for sha in commits if sha in known}
    new = [sha for sha in commits if sha not in known]
    if new:
        ids.update(git_util.git_patch_ids(repo_root, new))
        db.write(db.root / LANDED_DOC, ids)
    logger.debug("landed patch-ids: %d commits since %s, %d newly hashed",
                 len(commits), since.date(), len(new))
    return commits, ids
