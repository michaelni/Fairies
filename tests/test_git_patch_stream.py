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
git_remote_default_branch: the branch a remote's HEAD symref names."""
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


if __name__ == "__main__":
    unittest.main()
