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

Tests for containers/provision_remote.py.

All podman/git/ssh calls are mocked. These lock down the ssh-only
orchestration: the reachability check (``ssh DEST podman info``) and its
actionable error, Containerfile-hash image freshness, and mirror
seeding. No real podman, git, ssh, or network is touched.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "containers")):
    if p not in sys.path:
        sys.path.insert(0, p)

import podman_host as lc  # noqa: E402
import provision_remote as pr  # noqa: E402


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> mock.Mock:
    cp = mock.Mock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


class CheckReachableTests(unittest.TestCase):
    def test_ok_runs_podman_info_over_ssh(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(pr.subprocess, "run",
                               return_value=_completed(0, stdout="amd64")) as run:
            pr.check_reachable(host)
        self.assertEqual(host.argv(["podman", "info", "--format", "{{.Host.Arch}}"]),
                         run.call_args.args[0])

    def test_unreachable_gives_actionable_hint(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(pr.subprocess, "run",
                               return_value=_completed(125, stderr="cannot connect")):
            with self.assertRaisesRegex(RuntimeError, "passwordless ssh"):
                pr.check_reachable(host)


class EnsureImageTests(unittest.TestCase):
    def _df(self, tmp: Path) -> Path:
        df = tmp / "Containerfile"
        df.write_bytes(b"FROM scratch\n")
        return df

    def test_skips_when_label_matches(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with tempfile.TemporaryDirectory() as t:
            df = self._df(Path(t))
            want = pr.containerfile_sha256(df)
            with mock.patch.object(pr, "image_label", return_value=want) as label, \
                    mock.patch.object(pr, "build_image_if_needed") as build:
                pr.ensure_image(host=host, tag="t", dockerfile=df,
                                context_dir=Path(t), rebuild=False)
            self.assertEqual(host, label.call_args.kwargs["host"])
            self.assertFalse(build.call_args.kwargs["force"])
            self.assertEqual(host, build.call_args.kwargs["host"])
            self.assertEqual(want, build.call_args.kwargs["labels"][pr.CONTAINERFILE_LABEL])

    def test_forces_rebuild_when_label_stale(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with tempfile.TemporaryDirectory() as t:
            df = self._df(Path(t))
            with mock.patch.object(pr, "image_label", return_value="oldhash"), \
                    mock.patch.object(pr, "build_image_if_needed") as build:
                pr.ensure_image(host=host, tag="t", dockerfile=df,
                                context_dir=Path(t), rebuild=False)
            self.assertTrue(build.call_args.kwargs["force"])


class SeedMirrorsTests(unittest.TestCase):
    def test_seeds_each_repo(self) -> None:
        host = lc.RemoteHost("fairy@h")
        roots = [Path("/srv/ffmpeg"), Path("/srv/all_ffmpeg")]
        specs = [
            pr.podman_repos.RepoSpec(repo_root=roots[0], name="ffmpeg", head_sha="a" * 40,
                                    container_path="/work/ffmpeg",
                                    mirror_path="fairy-mirrors/ffmpeg.git"),
            pr.podman_repos.RepoSpec(repo_root=roots[1], name="all_ffmpeg", head_sha="b" * 40,
                                    container_path="/work/all_ffmpeg",
                                    mirror_path="fairy-mirrors/all_ffmpeg.git"),
        ]
        with mock.patch.object(pr.podman_repos, "build_repo_specs", return_value=specs), \
                mock.patch.object(pr.podman_repos, "ensure_remote_mirror") as ensure, \
                mock.patch.object(pr.podman_repos, "sync_repo_to_mirror") as sync:
            pr.seed_mirrors(roots, host, "fairy-mirrors")
        self.assertEqual(
            ["fairy-mirrors/ffmpeg.git", "fairy-mirrors/all_ffmpeg.git"],
            [c.args[1] for c in ensure.call_args_list],
        )
        self.assertEqual([specs[0], specs[1]], [c.args[0] for c in sync.call_args_list])


if __name__ == "__main__":
    unittest.main()
