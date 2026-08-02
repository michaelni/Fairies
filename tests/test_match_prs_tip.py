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

match_prs_to_master must derive PR status from the real head ref.

The union walk (``git log --stdin --not base``) date-sorts its output,
so an ancestor with a skewed (newer) commit date that is shared with
another PR can be emitted before this PR's head. ``divergent[0]`` is
then not the head; taking it as the tip reported a merged PR as open.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "match_prs_to_master.py"


def _git(repo: Path, *args: str, date: str = "2026-01-01T12:00:00+0000") -> str:
    env = {
        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@x",
        "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, check=True, text=True, env=env,
    ).stdout


@unittest.skipUnless(shutil.which("git"), "git required")
class TipIsHeadTests(unittest.TestCase):
    def test_skewed_shared_ancestor_does_not_displace_tip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="match-prs-") as tmp:
            repo = Path(tmp)
            _git(repo, "init", "--initial-branch=master", "--quiet")
            (repo / "f").write_text("0\n")
            _git(repo, "add", "f")
            _git(repo, "commit", "-qm", "M0")

            # Shared ancestor with a commit date NEWER than pr/1's head
            # (clock skew); reachable from both PRs so the union walk
            # emits it, date-first, before pr/1's head.
            _git(repo, "checkout", "-qb", "shared")
            (repo / "x").write_text("x\n")
            _git(repo, "add", "x")
            _git(repo, "commit", "-qm", "shared work",
                 date="2026-01-05T12:00:00+0000")

            _git(repo, "checkout", "-qb", "pr1")
            (repo / "f").write_text("1\n")
            _git(repo, "commit", "-aqm", "fix bug",
                 date="2026-01-02T12:00:00+0000")
            head1 = _git(repo, "rev-parse", "pr1").strip()

            _git(repo, "checkout", "-qb", "pr2", "shared")
            (repo / "g").write_text("g\n")
            _git(repo, "add", "g")
            _git(repo, "commit", "-qm", "other work",
                 date="2026-01-06T12:00:00+0000")

            # pr/1's head lands on master via cherry-pick (author date
            # kept, new sha): the PR is merged.
            _git(repo, "checkout", "-q", "master")
            _git(repo, "cherry-pick", head1, date="2026-01-07T12:00:00+0000")

            _git(repo, "update-ref", "refs/remotes/fforge/pr/1", "pr1")
            _git(repo, "update-ref", "refs/remotes/fforge/pr/2", "pr2")

            out = subprocess.run(
                [sys.executable, str(SCRIPT), "--base", "master",
                 "--format", "tsv", "--color", "never"],
                cwd=repo, capture_output=True, text=True, check=True,
            ).stdout
            row1 = next(l.split("\t") for l in out.splitlines()
                        if l.startswith("fforge/pr/1\t"))
            self.assertEqual(row1[2], head1[:12])
            self.assertEqual(row1[1], "merged")


if __name__ == "__main__":
    unittest.main()
