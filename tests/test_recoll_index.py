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

The container image ships a recoll setup matching what the prompt
advertises: recollcmd installed, /root/.recoll/recoll.conf indexing
/work with the duplicate forgejo_git .json exports skipped, and a
blocking recollq wrapper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import podman_host  # noqa: E402
import podman_repos  # noqa: E402

CONTAINERFILE = (REPO_ROOT / "containers" / "Containerfile").read_text()
RECOLL_CONF = (REPO_ROOT / "containers" / "recoll.conf").read_text()
RECOLLQ_WRAPPER = REPO_ROOT / "containers" / "recollq"


class ContainerfileRecollTests(unittest.TestCase):
    def test_installs_recollcmd_and_doc_filters(self) -> None:
        self.assertIn("recollcmd", CONTAINERFILE)
        self.assertIn("antiword", CONTAINERFILE)

    def test_copies_conf_and_wrapper(self) -> None:
        self.assertIn("COPY recoll.conf mimemap /root/.recoll/", CONTAINERFILE)
        self.assertIn("COPY recollq /usr/local/bin/recollq", CONTAINERFILE)

    def test_mimemap_makes_texinfo_plain_text(self) -> None:
        mimemap = (REPO_ROOT / "containers" / "mimemap").read_text()
        self.assertIn(".texi = text/plain", mimemap)


class RecollConfTests(unittest.TestCase):
    def test_indexes_work_and_skips_git(self) -> None:
        self.assertIn("topdirs = /work", RECOLL_CONF)
        self.assertIn("skippedNames+ = .git", RECOLL_CONF)

    def test_types_suffixless_sources(self) -> None:
        # without these, suffixless sources are not indexed (no xdg-mime in the image)
        self.assertIn("systemfilecommand = file -b --mime-type", RECOLL_CONF)
        self.assertIn("textunknownasplain = 1", RECOLL_CONF)

    def test_forgejo_git_json_duplicates_skipped(self) -> None:
        # issues/PRs are exported as .md + .json pairs; only the .md is indexed
        for section in ("[/work/all_ffmpeg/forgejo_git]", "[/work/forgejo_git]"):
            self.assertIn(f"{section}\nskippedNames+ = *.json", RECOLL_CONF)


class RecollqWrapperTests(unittest.TestCase):
    def test_blocks_on_marker_then_execs_real_recollq(self) -> None:
        text = RECOLLQ_WRAPPER.read_text()
        # same marker start_recoll_index touches when indexing finished
        self.assertIn("/root/.recoll/index.done", text)
        self.assertIn('exec /usr/bin/recollq "$@"', text)
        self.assertTrue(RECOLLQ_WRAPPER.stat().st_mode & 0o111,
                        "wrapper must be executable (COPY keeps the mode)")


class StartRecollIndexTests(unittest.TestCase):
    HANDLE = podman_host.ContainerHandle(
        "cid42", "img", None, podman_host.RemoteHost("fairy@h"))

    def test_detached_exec_builds_index_and_marker(self) -> None:
        ok = mock.Mock(returncode=0, stdout=b"", stderr=b"")
        with mock.patch.object(podman_repos, "run_on_remote_host",
                               return_value=ok) as r:
            podman_repos.start_recoll_index(self.HANDLE)
        argv = r.call_args[0]
        self.assertEqual(("podman", "exec", "-d", "cid42", "sh", "-c"), argv[1:7])
        self.assertIn("recollindex", argv[7])
        # marker the recollq wrapper blocks on
        self.assertIn("touch /root/.recoll/index.done", argv[7])

    def test_kickoff_failure_does_not_raise(self) -> None:
        boom = mock.Mock(returncode=1, stdout=b"", stderr=b"no recollindex")
        with mock.patch.object(podman_repos, "run_on_remote_host",
                               return_value=boom):
            podman_repos.start_recoll_index(self.HANDLE)


if __name__ == "__main__":
    unittest.main()
