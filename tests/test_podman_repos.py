"""Tests for the vendor-agnostic ``podman_repos`` module.

All ``podman``/``git``/ssh invocations are mocked. The point is to lock
down (a) the name sanitisation + dedup algorithm matching the existing
OpenAI-side layout, and (b) provisioning: mirror path derivation,
``ensure_remote_mirror``, the thin ``git push`` sync, and the
host-local cp/checkout sequence. None of these tests need real podman,
git, or ssh on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import podman_host as lc  # noqa: E402
import podman_repos as lr  # noqa: E402


def _completed(returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> mock.Mock:
    cp = mock.Mock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


class BuildRepoSpecsTests(unittest.TestCase):
    def test_sanitises_name_and_appends_to_container_root(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="deadbeef" * 5):
            specs = lr.build_repo_specs(
                [Path("/srv/messy name!.weird/")], container_root="/work",
            )
        self.assertEqual(1, len(specs))
        self.assertEqual("messy-name-.weird", specs[0].name)
        self.assertEqual("/work/messy-name-.weird", specs[0].container_path)

    def test_dedups_collisions_with_numeric_suffix(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="0" * 40):
            specs = lr.build_repo_specs(
                [Path("/a/ffmpeg"), Path("/b/ffmpeg"), Path("/c/ffmpeg")],
            )
        self.assertEqual(["ffmpeg", "ffmpeg-2", "ffmpeg-3"], [s.name for s in specs])

    def test_default_container_root_is_slash_work(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="x" * 40):
            specs = lr.build_repo_specs([Path("/srv/ffmpeg")])
        self.assertEqual("/work/ffmpeg", specs[0].container_path)

    def test_records_head_sha_from_get_repo_head_sha(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="cafebabe" * 5):
            specs = lr.build_repo_specs([Path("/srv/ffmpeg")])
        self.assertEqual("cafebabe" * 5, specs[0].head_sha)

    def test_empty_or_unprintable_name_falls_back_to_repo(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="0" * 40):
            specs = lr.build_repo_specs([Path("/srv/!!!")])
        self.assertEqual("repo", specs[0].name)

    def test_mirror_path_uses_default_root(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="0" * 40):
            specs = lr.build_repo_specs([Path("/srv/ffmpeg")])
        self.assertEqual(f"{lr.DEFAULT_MIRROR_ROOT}/ffmpeg.git", specs[0].mirror_path)

    def test_mirror_path_built_from_mirror_root(self) -> None:
        with mock.patch.object(lr, "get_repo_head_sha", return_value="0" * 40):
            specs = lr.build_repo_specs(
                [Path("/a/ffmpeg"), Path("/b/ffmpeg")], mirror_root="fairy-mirrors/",
            )
        self.assertEqual(
            ["fairy-mirrors/ffmpeg.git", "fairy-mirrors/ffmpeg-2.git"],
            [s.mirror_path for s in specs],
        )


class RemoteProvisionTests(unittest.TestCase):
    def _remote_spec(self) -> lr.RepoSpec:
        return lr.RepoSpec(
            repo_root=Path("/srv/ffmpeg"), name="ffmpeg",
            head_sha="a" * 40, container_path="/work/ffmpeg",
            mirror_path="fairy-mirrors/ffmpeg.git",
        )

    def test_ensure_remote_mirror_mkdir_then_init(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(lr, "run_on_remote_host", return_value=_completed(0)) as r:
            lr.ensure_remote_mirror(host, "fairy-mirrors/ffmpeg.git")
        self.assertEqual(
            [(host, "mkdir", "-p", "fairy-mirrors"),
             (host, "git", "init", "--bare", "--quiet", "fairy-mirrors/ffmpeg.git")],
            [c.args for c in r.call_args_list],
        )

    # Real failure observed 2026-07-10 with 3 parallel issue reviews:
    # ``git init --bare`` on an existing mirror still takes the config
    # lock, so the loser of the race fails rc=128.
    _INIT_LOCK_STDERR = (
        b"error: could not lock config file "
        b"/home/fairy/fairy-mirrors/ffmpeg.git/config: File exists"
    )

    def test_ensure_remote_mirror_retries_config_lock_race(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(lr, "run_on_remote_host", side_effect=[
            _completed(0),  # mkdir
            _completed(128, stderr=self._INIT_LOCK_STDERR),
            _completed(0),
        ]) as r, mock.patch.object(lr.time, "sleep") as slept:
            lr.ensure_remote_mirror(host, "fairy-mirrors/ffmpeg.git")
        self.assertEqual(3, r.call_count)
        self.assertTrue(0.0 < slept.call_args.args[0] <= 3.0)

    def test_ensure_remote_mirror_raises_after_max_attempts(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(lr, "run_on_remote_host", side_effect=[
            _completed(0),
            *[_completed(128, stderr=self._INIT_LOCK_STDERR)] * 3,
        ]), mock.patch.object(lr.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "could not lock config file"):
                lr.ensure_remote_mirror(host, "fairy-mirrors/ffmpeg.git")

    def test_sync_pushes_head_ref_and_all_refs_with_ssh_command(self) -> None:
        host = lc.RemoteHost("fairy@h", identity="/k/id", port=17022)
        with mock.patch.object(lr, "git_push_refspecs") as push:
            lr.sync_repo_to_mirror(self._remote_spec(), host)
        args, kwargs = push.call_args
        self.assertEqual(Path("/srv/ffmpeg"), args[0])
        self.assertEqual("fairy@h:fairy-mirrors/ffmpeg.git", args[1])
        # The head-SHA refspec plus the complete-refs refspec: the mirror
        # (and thus the container) carries every client ref, notably the
        # fforge/pr/* PR heads reviewers inspect.
        self.assertEqual(
            [f"{'a' * 40}:refs/fairy/heads/ffmpeg", "refs/*:refs/*"],
            args[2],
        )
        self.assertEqual(
            "ssh -o BatchMode=yes -o ServerAliveInterval=30 "
            "-o ServerAliveCountMax=3 -o LogLevel=ERROR -i /k/id -p 17022",
            kwargs["ssh_command"],
        )

    # Real push failure observed 2026-07-02 when two concurrent reviews
    # synced the same head to the same mirror ref; the loser must retry
    # rather than fail the review.
    _LOST_RACE_ERROR = RuntimeError(
        "git push 08f56d4898eafcaddc19b3aac9263e066e82f0c0->refs/fairy/heads/ffmpeg "
        "to fairy@h:fairy-mirrors/ffmpeg.git from "
        "/home/ai/cursor/forgejo_fairy/simpast-mirror/ffmpeg failed: "
        "remote: error: cannot lock ref 'refs/fairy/heads/ffmpeg': "
        "is at 08f56d4898eafcaddc19b3aac9263e066e82f0c0 but expected "
        "3e2ebba1c421b1a63d7cebcb27e1a0c93d1d1638"
    )

    def test_sync_lost_race_retries_and_succeeds(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(lr, "git_push_refspecs",
                               side_effect=[self._LOST_RACE_ERROR, None]) as push, \
                mock.patch.object(lr.time, "sleep") as slept:
            lr.sync_repo_to_mirror(self._remote_spec(), host)
        self.assertEqual(2, push.call_count)
        delay = slept.call_args.args[0]
        self.assertTrue(0.0 < delay <= 3.0)

    def test_sync_push_failure_raises_after_max_attempts(self) -> None:
        host = lc.RemoteHost("fairy@h")
        with mock.patch.object(lr, "git_push_refspecs",
                               side_effect=self._LOST_RACE_ERROR) as push, \
                mock.patch.object(lr.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "cannot lock ref"):
                lr.sync_repo_to_mirror(self._remote_spec(), host)
        self.assertEqual(3, push.call_count)

    def test_provision_prunes_future_refs_when_cutoff_given(self) -> None:
        # Simulate-past: the mirror carries today's refs; refs whose commits
        # postdate the cutoff must be deleted in the container so
        # ``git log --all`` cannot see the future (past force-pushes stay).
        host = lc.RemoteHost("fairy@h")
        handle = lc.ContainerHandle(
            container_id="cid", image="img", network=None, host=host,
        )
        spec = self._remote_spec()
        with mock.patch.object(lr, "ensure_remote_mirror"), \
                mock.patch.object(lr, "sync_repo_to_mirror"), \
                mock.patch.object(lr, "run_on_remote_host", return_value=_completed(0)) as r:
            lr.provision_repos_into_container(handle, [spec], host,
                                              prune_refs_after=1776974400)
        prune = [c.args for c in r.call_args_list if "sh" in c.args][-1]
        joined = " ".join(a for a in prune if isinstance(a, str))
        self.assertIn("for-each-ref", joined)
        self.assertIn("1776974400", joined)
        self.assertIn("update-ref --stdin", joined)
        self.assertIn('"delete ', joined)
        # Without a cutoff no prune step runs.
        with mock.patch.object(lr, "ensure_remote_mirror"), \
                mock.patch.object(lr, "sync_repo_to_mirror"), \
                mock.patch.object(lr, "run_on_remote_host", return_value=_completed(0)) as r:
            lr.provision_repos_into_container(handle, [spec], host)
        self.assertFalse([c for c in r.call_args_list
                          if "for-each-ref" in " ".join(a for a in c.args if isinstance(a, str))])

    def test_remote_provision_sequence(self) -> None:
        host = lc.RemoteHost("fairy@h")
        handle = lc.ContainerHandle(
            container_id="cid", image="img", network=None, host=host,
        )
        spec = self._remote_spec()
        with mock.patch.object(lr, "ensure_remote_mirror") as ensure, \
                mock.patch.object(lr, "sync_repo_to_mirror") as sync, \
                mock.patch.object(lr, "run_on_remote_host", return_value=_completed(0)) as r:
            lr.provision_repos_into_container(handle, [spec], host)
        ensure.assert_called_once_with(host, "fairy-mirrors/ffmpeg.git")
        sync.assert_called_once_with(spec, host)
        # All four podman steps run host-locally over ssh, in order.
        self.assertEqual(
            [(host, "podman", "exec", "cid", "mkdir", "-p", "/work/ffmpeg"),
             (host, "podman", "cp", "fairy-mirrors/ffmpeg.git", "cid:/work/ffmpeg/.git"),
             (host, "podman", "exec", "cid", "git", "-C", "/work/ffmpeg",
              "config", "core.bare", "false"),
             (host, "podman", "exec", "cid", "git", "-C", "/work/ffmpeg",
              "reset", "--hard", "a" * 40)],
            [c.args for c in r.call_args_list],
        )

    def test_multiple_repos_provisioned_in_order(self) -> None:
        host = lc.RemoteHost("fairy@h")
        handle = lc.ContainerHandle(container_id="cid", image="img", network=None,
                                    host=host)
        specs = [
            lr.RepoSpec(Path("/srv/a"), "a", "a" * 40, "/work/a", "fairy-mirrors/a.git"),
            lr.RepoSpec(Path("/srv/b"), "b", "b" * 40, "/work/b", "fairy-mirrors/b.git"),
        ]
        with mock.patch.object(lr, "ensure_remote_mirror"), \
                mock.patch.object(lr, "sync_repo_to_mirror"), \
                mock.patch.object(lr, "run_on_remote_host", return_value=_completed(0)) as r:
            lr.provision_repos_into_container(handle, specs, host)
        mkdirs = [c.args for c in r.call_args_list if c.args[2] == "exec" and c.args[4] == "mkdir"]
        self.assertEqual(["/work/a", "/work/b"], [c[6] for c in mkdirs])

    def test_remote_provision_raises_on_remote_failure(self) -> None:
        host = lc.RemoteHost("fairy@h")
        handle = lc.ContainerHandle(
            container_id="cid", image="img", network=None, host=host,
        )
        with mock.patch.object(lr, "ensure_remote_mirror"), \
                mock.patch.object(lr, "sync_repo_to_mirror"), \
                mock.patch.object(lr, "run_on_remote_host",
                                  return_value=_completed(1, stderr=b"no such container")):
            with self.assertRaisesRegex(RuntimeError, "remote-local"):
                lr.provision_repos_into_container(handle, [self._remote_spec()], host)


@unittest.skipUnless(shutil.which("git"), "git required")
class FullRefsSyncTests(unittest.TestCase):
    """Real-git check that the sync refspecs carry PR heads.

    Regression for the 2026-07-02 ensemble run where reviewers could not
    inspect PR head 1ce2a4db... inside /work/ffmpeg: only the review head
    had been pushed to the mirror, so commits reachable solely from
    ``refs/remotes/fforge/pr/*`` were missing from the container.
    """

    def test_pr_ref_unreachable_from_head_lands_in_mirror(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fairy-fullsync-") as tmp:
            repo = Path(tmp) / "src"
            repo.mkdir()

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", "-C", str(repo), "-c", "user.email=t@t",
                     "-c", "user.name=t", *args],
                    check=True, capture_output=True, text=True,
                ).stdout

            git("init", "--quiet")
            (repo / "f").write_text("x\n")
            git("add", "f")
            git("commit", "-m", "base", "--quiet")
            base = git("rev-parse", "HEAD").strip()
            (repo / "g").write_text("y\n")
            git("add", "g")
            git("commit", "-m", "pr head", "--quiet")
            pr_head = git("rev-parse", "HEAD").strip()
            # Leave the PR commit reachable only from the fforge PR ref,
            # exactly like an unmerged PR in the client ffmpeg repo.
            git("update-ref", "refs/remotes/fforge/pr/123", pr_head)
            git("reset", "--hard", base, "--quiet")

            mirror = Path(tmp) / "mirror.git"
            subprocess.run(["git", "init", "--bare", "--quiet", str(mirror)], check=True)
            from git_util import git_push_refspecs
            git_push_refspecs(
                repo, str(mirror),
                [f"{base}:refs/fairy/heads/src", "refs/*:refs/*"],
            )

            mirror_pr_sha = subprocess.run(
                ["git", "--git-dir", str(mirror),
                 "rev-parse", "refs/remotes/fforge/pr/123"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            self.assertEqual(pr_head, mirror_pr_sha)


if __name__ == "__main__":
    unittest.main()
