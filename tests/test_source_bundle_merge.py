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

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pr_review_wrapper as wrapper  # noqa: E402


BASE = "".join(f"line{i}\n" for i in range(20))
FIRST_SHA = "3bd4c4373a7d36e228b6a28566fa3c43f9aa6152"
HEAD_SHA = "a14bf7a966addf29772e106a0a54b94af3289c53"
MERGE_BASE = "2209a453c8fc298fb94c894b2ac6b52fa22f3bcc"
TARGET_TIP = "d606e7fa85132dbb2ca5b1e85c8aefa35c44127d"
MERGED_TREE = "d2ccaf859e42c58d44a928ea355770f121ea38d1"
PR_F_C = BASE.replace("line2\n", "pr_change\n").replace("line5\n", "second_pr_change\n")
REPO_FILES = {
    (FIRST_SHA, "f.c"): BASE.replace("line2\n", "pr_change\n"),
    (HEAD_SHA, "f.c"): PR_F_C,
    (MERGED_TREE, "f.c"): PR_F_C.replace("line17\n", "master_change\n"),
}
PATCH = """\
From 3bd4c4373a7d36e228b6a28566fa3c43f9aa6152 Mon Sep 17 00:00:00 2001
From: T <t@x>
Date: Fri, 1 May 2026 12:00:00 +0000
Subject: [PATCH 1/2] pr work

---
 f.c | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/f.c b/f.c
index 0bbfb24..6bb9624 100644
--- a/f.c
+++ b/f.c
@@ -1,6 +1,6 @@
 line0
 line1
-line2
+pr_change
 line3
 line4
 line5
--\x20
2.43.0


From a14bf7a966addf29772e106a0a54b94af3289c53 Mon Sep 17 00:00:00 2001
From: T <t@x>
Date: Fri, 1 May 2026 12:00:00 +0000
Subject: [PATCH 2/2] more pr work

---
 f.c | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/f.c b/f.c
index 6bb9624..03dd813 100644
--- a/f.c
+++ b/f.c
@@ -3,7 +3,7 @@ line1
 pr_change
 line3
 line4
-line5
+second_pr_change
 line6
 line7
 line8
--\x20
2.43.0

"""


class SourceBundleMergeTests(unittest.TestCase):
    def _bundle(self, merged_tree: str | None | RuntimeError = MERGED_TREE,
                patch: str = PATCH) -> tuple[str, list[str], list[str]]:
        request = {
            "pull_request": {"base_sha": MERGE_BASE, "head_sha": HEAD_SHA,
                             "base_ref": "master"},
            "patch": patch,
        }
        with (
            mock.patch.object(wrapper, "git_changed_paths", return_value=["f.c"]),
            mock.patch.object(wrapper, "resolve_target_tip", return_value=TARGET_TIP),
            mock.patch.object(wrapper, "git_merge_tree", side_effect=[merged_tree]),
            mock.patch.object(wrapper, "git_show_file",
                              lambda _repo, revision, relpath: REPO_FILES.get((revision, relpath))),
        ):
            return wrapper.build_source_bundle(
                request, Path("repo"),
                max_source_files=5, max_file_bytes=10000,
                max_header_file_bytes=10000, max_bundle_bytes=100000,
                include_direct_includes=False, verbose=False,
            )

    def test_bundle_contains_merge_of_head_and_target_tip(self) -> None:
        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)
        self.assertIn("second_pr_change", bundle)
        self.assertIn("master_change", bundle)
        self.assertIn("merge result of the PR head", bundle)
        self.assertIn(f"merge of PR head {HEAD_SHA} into master", bundle)

    def test_conflicting_merge_falls_back_to_pr_commits(self) -> None:
        bundle, used, notes = self._bundle(merged_tree=None)
        self.assertEqual(["f.c"], used)
        self.assertIn("second_pr_change", bundle)
        self.assertNotIn("master_change", bundle)
        self.assertIn(f"git show {HEAD_SHA}:f.c", bundle)
        self.assertTrue(any("does not merge cleanly" in n for n in notes))

    def test_failed_merge_falls_back_without_claiming_a_conflict(self) -> None:
        bundle, used, notes = self._bundle(merged_tree=RuntimeError("git merge-tree failed"))
        self.assertIn(f"git show {HEAD_SHA}:f.c", bundle)
        self.assertFalse(any("does not merge cleanly" in n for n in notes))

    def test_patch_cut_inside_a_diff_header(self) -> None:
        bundle, used, notes = self._bundle(patch=PATCH[:PATCH.index(" b/f.c") + len(" b/f")])
        self.assertEqual(["f.c"], used)


if __name__ == "__main__":
    unittest.main()
