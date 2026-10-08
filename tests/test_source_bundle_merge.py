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

Source-bundle files are the PR's predicted merge result.

Sending PR-head file contents while the vector store / container
checkout sit at the target tip made models (gpt-5.4, glm-5.2,
gpt-5.6-terra on PR #23720) conclude the PR "removes" newer target
work. The bundle must therefore contain the merge of the PR head into
the current target tip, labeled as such, and fall back to PR-commit
contents with a note when the merge conflicts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pr_review_wrapper as wrapper  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@x",
        "GIT_AUTHOR_DATE": "2026-05-01T12:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-05-01T12:00:00+00:00",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, check=True, text=True, env=env,
    ).stdout


BASE = "".join(f"line{i}\n" for i in range(20))


@unittest.skipUnless(shutil.which("git"), "git required")
class SourceBundleMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pr-fairy-bundle-")
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "r"
        self.repo.mkdir()
        _git(self.repo, "init", "--initial-branch=master", "--quiet")
        (self.repo / "f.c").write_text(BASE)
        (self.repo / "gone.c").write_text(BASE)
        (self.repo / "sub").mkdir()
        _git(self.repo, "add", "f.c", "gone.c")
        _git(self.repo, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},sub")
        _git(self.repo, "commit", "-m", "base", "--quiet")
        self.base_sha = _git(self.repo, "rev-parse", "HEAD").strip()
        _git(self.repo, "branch", "pr")
        self._pr_commit("line2\n", "pr_change\n")

    def _pr_commit(self, old_line: str, new_line: str) -> None:
        _git(self.repo, "checkout", "pr", "--quiet")
        f_c = self.repo / "f.c"
        f_c.write_text(f_c.read_text().replace(old_line, new_line))
        _git(self.repo, "add", "f.c")
        _git(self.repo, "commit", "-m", "pr work", "--quiet")
        self.head_sha = _git(self.repo, "rev-parse", "HEAD").strip()
        self.patch = _git(self.repo, "format-patch", "--stdout", "master..pr")
        _git(self.repo, "checkout", "--force", "master", "--quiet")

    def _bundle(self) -> tuple[str, list[str], list[str]]:
        request = {
            "pull_request": {"base_sha": self.base_sha, "head_sha": self.head_sha,
                             "base_ref": "master"},
            "patch": self.patch,
        }
        return wrapper.build_source_bundle(
            request, self.repo,
            max_source_files=5, max_file_bytes=10000,
            max_header_file_bytes=10000, max_bundle_bytes=100000,
            include_direct_includes=False, verbose=False,
        )

    def test_bundle_contains_merge_of_head_and_target_tip(self) -> None:
        # master moved on after the merge base, touching a distant line
        (self.repo / "f.c").write_text(BASE.replace("line17\n", "master_change\n"))
        _git(self.repo, "commit", "-am", "master work", "--quiet")

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)
        self.assertIn("pr_change", bundle)
        self.assertIn("master_change", bundle)
        self.assertIn("merge result of the PR head", bundle)
        self.assertIn(f"merge of PR head {self.head_sha} into master", bundle)

    def test_conflicting_merge_falls_back_to_pr_commits(self) -> None:
        # master rewrote the same line the PR changes: merge conflicts
        (self.repo / "f.c").write_text(BASE.replace("line2\n", "master_conflict\n"))
        _git(self.repo, "commit", "-am", "master work", "--quiet")
        self._pr_commit("line5\n", "second_pr_change\n")

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)
        self.assertIn("pr_change", bundle)
        self.assertIn("second_pr_change", bundle)
        self.assertNotIn("master_conflict", bundle)
        self.assertIn(f"git show {self.head_sha}:f.c", bundle)
        self.assertTrue(any("does not merge cleanly" in n for n in notes))

    def test_patch_cut_inside_a_diff_header(self) -> None:
        self.patch = self.patch[:self.patch.index(" b/f.c") + len(" b/f")]

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)

    def test_deleted_files_and_submodules_are_left_out(self) -> None:
        _git(self.repo, "checkout", "pr", "--quiet")
        _git(self.repo, "rm", "--quiet", "gone.c")
        _git(self.repo, "update-index", "--add", "--cacheinfo", f"160000,{self.base_sha},ext")
        self._pr_commit("line9\n", "pr_change_next_to_removals\n")

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)

    def test_file_and_submodule_swapping_type(self) -> None:
        gone_c_blob = _git(self.repo, "rev-parse", f"{self.base_sha}:gone.c").strip()
        _git(self.repo, "checkout", "pr", "--quiet")
        _git(self.repo, "rm", "--quiet", "--cached", "gone.c", "sub")
        _git(self.repo, "update-index", "--add", "--cacheinfo", f"160000,{self.base_sha},gone.c")
        _git(self.repo, "update-index", "--add", "--cacheinfo", f"100644,{gone_c_blob},sub")
        self._pr_commit("line9\n", "pr_change_next_to_type_changes\n")

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c", "sub"], used)

    def test_base_sha_may_be_the_target_tip(self) -> None:
        (self.repo / "gone.c").write_text("master_change\n")
        _git(self.repo, "commit", "-am", "master work", "--quiet")
        self.base_sha = _git(self.repo, "rev-parse", "HEAD").strip()

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)


if __name__ == "__main__":
    unittest.main()
