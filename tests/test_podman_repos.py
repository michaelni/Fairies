"""Tests for the vendor-agnostic ``podman_repos`` module.

All ``podman``/``git``/ssh invocations are mocked. The point is to lock
down (a) the name sanitisation + dedup algorithm matching the existing
OpenAI-side layout, and (b) provisioning: mirror path derivation,
``ensure_remote_mirror``, the thin ``git push`` sync, and the
host-local cp/checkout sequence. None of these tests need real podman,
git, or ssh on PATH.
"""

from __future__ import annotations

import sys
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

    def test_sync_pushes_head_with_ssh_command_and_ref(self) -> None:
        host = lc.RemoteHost("fairy@h", identity="/k/id")
        with mock.patch.object(lr, "git_push_commit") as push:
            lr.sync_repo_to_mirror(self._remote_spec(), host)
        args, kwargs = push.call_args
        self.assertEqual(Path("/srv/ffmpeg"), args[0])
        self.assertEqual("fairy@h:fairy-mirrors/ffmpeg.git", args[1])
        self.assertEqual("a" * 40, args[2])
        self.assertEqual("refs/fairy/heads/ffmpeg", args[3])
        self.assertEqual(
            "ssh -o BatchMode=yes -o ServerAliveInterval=30 "
            "-o ServerAliveCountMax=3 -i /k/id",
            kwargs["ssh_command"],
        )

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


if __name__ == "__main__":
    unittest.main()
