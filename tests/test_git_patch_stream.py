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

git_patch_stream: format-patch per non-merge commit, git log per merge;
git_remote_default_branch: the branch a remote's HEAD symref names;
git_merge_tree: the merged tree, None on a conflict;
git_changed_paths: the files the head leaves in the tree."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import git_util

SHA_A, SHA_MERGE, PARENT = "a" * 40, "b" * 40, "c" * 40


class PatchStreamTests(unittest.TestCase):
    def test_merges_ride_as_log_between_the_format_patches(self) -> None:
        calls = []

        def fake_git(repo_root, *args):
            calls.append(args)
            if args[0] == "rev-list":
                return (f"{SHA_A} {PARENT}\n"
                        f"{SHA_MERGE} {PARENT} {SHA_A}\n").encode()
            return b"PATCH-A " if args[0] == "format-patch" else b"LOG-MERGE"

        with mock.patch.object(git_util, "_git_stdout", fake_git):
            out = git_util.git_patch_stream(Path("repo"), "base", "head")
        self.assertEqual(out, b"PATCH-A LOG-MERGE")
        self.assertEqual(calls[0][:2], ("rev-list", "--reverse"))
        self.assertIn("base..head", calls[0])
        self.assertEqual([(c[0], c[-1]) for c in calls[1:]],
                         [("format-patch", SHA_A), ("log", SHA_MERGE)])

    def test_no_base_streams_the_whole_history(self) -> None:
        with mock.patch.object(git_util, "_git_stdout",
                               return_value=b"") as git:
            git_util.git_patch_stream(Path("repo"), None, "head")
        self.assertEqual(git.call_args.args[-1], "head")

    def test_an_empty_range_yields_no_bytes_and_no_per_commit_calls(
            self) -> None:
        with mock.patch.object(git_util, "_git_stdout",
                               return_value=b"") as git:
            self.assertEqual(
                git_util.git_patch_stream(Path("repo"), "base", "head"), b"")
        git.assert_called_once()


class RemoteDefaultBranchTests(unittest.TestCase):
    LISTING = ("ref: refs/heads/master\tHEAD\n"
               "5fb7f9c5deee9fb4b580b20efe150be300bed6f8\tHEAD\n")

    def test_the_head_symref_names_the_branch(self) -> None:
        with mock.patch.object(git_util.subprocess, "run", return_value=mock.Mock(
                returncode=0, stdout=self.LISTING, stderr="")) as run:
            branch = git_util.git_remote_default_branch(
                Path("repo"), "https://forge.example.com/f/r.git")
        self.assertEqual(branch, "master")
        self.assertEqual(run.call_args.args[0][3:],
                         ["ls-remote", "--symref",
                          "https://forge.example.com/f/r.git", "HEAD"])

    def test_a_listing_without_the_symref_raises(self) -> None:
        with mock.patch.object(git_util.subprocess, "run", return_value=mock.Mock(
                returncode=0, stdout=self.LISTING.splitlines()[1], stderr="")):
            with self.assertRaisesRegex(RuntimeError, "no HEAD branch"):
                git_util.git_remote_default_branch(Path("repo"), "u")


DIFF_TREE = (
    b":160000 160000 2222222222222222222222222222222222222222 5555555555555555555555555555555555555555 M\0bump\0"
    b":100644 000000 286c5f5776916d7d7d5849988ca9d83e722cf9c2 0000000000000000000000000000000000000000 D\0del.c\0"
    b":000000 160000 0000000000000000000000000000000000000000 4444444444444444444444444444444444444444 A\0ext\0"
    b":100644 160000 f73f3093ff865c514c6c51f867e35f693487d0d3 6666666666666666666666666666666666666666 T\0f2s.c\0"
    b":100644 100644 df967b96a579e45a18b8251732d16804b2e56a55 5ea2ed416fbd4a4cbe227b75fe255dd7fa6bd4d6 M\0mod.c\0"
    b":000000 100644 0000000000000000000000000000000000000000 3e757656cf36eca53338e520d134963a44f793f8 A\0new file.c\0"
    b":160000 100644 1111111111111111111111111111111111111111 6ae26cfa7c81417efdd745bf21d1752b640c02df T\0s2f\0"
    b":160000 000000 3333333333333333333333333333333333333333 0000000000000000000000000000000000000000 D\0sub_del\0"
)


class TreeChangesTests(unittest.TestCase):
    """``DIFF_TREE`` is ``git diff-tree -r -z --merge-base`` output recorded
    in a test repository whose head modifies, adds and deletes files, adds,
    bumps and deletes submodules, turns a file into a submodule and a
    submodule into a file."""

    def test_changed_paths_leave_out_deletions_and_submodules(self) -> None:
        with mock.patch.object(git_util, "_git_stdout", return_value=DIFF_TREE) as git:
            paths = git_util.git_changed_paths(Path("repo"), "base", "head")
        self.assertEqual(paths, ["mod.c", "new file.c", "s2f"])
        self.assertIn("--merge-base", git.call_args.args)


class MergeTreeTests(unittest.TestCase):
    """Outputs of ``git merge-tree --write-tree`` recorded in a test repository."""

    def _merge_tree(self, returncode: int, stdout: str) -> str | None:
        with mock.patch.object(git_util.subprocess, "run", return_value=mock.Mock(
                returncode=returncode, stdout=stdout, stderr="")):
            return git_util.git_merge_tree(Path("repo"), "head", "tip")

    def test_a_clean_merge_yields_its_tree(self) -> None:
        self.assertEqual(self._merge_tree(0, "d2ccaf859e42c58d44a928ea355770f121ea38d1\n"),
                         "d2ccaf859e42c58d44a928ea355770f121ea38d1")

    def test_a_conflict_yields_none(self) -> None:
        self.assertIsNone(self._merge_tree(1, (
            "9c984ccab288b0a289043c493ae1e58a7580fc1c\n"
            "100644 0bbfb241ce76dab669e3fba205dd89ec6b85a8a8 1\tf.c\n"
            "100644 03dd8137299d30165c2081ce4ab59ea70ddc029f 2\tf.c\n"
            "100644 1a2d85c39ee79f163ab075a43900662257196aa7 3\tf.c\n"
            "\n"
            "Auto-merging f.c\n"
            "CONFLICT (content): Merge conflict in f.c\n")))


if __name__ == "__main__":
    unittest.main()
