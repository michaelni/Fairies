"""CodexContainer: ``podman exec`` argv construction over RemoteHost ssh."""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import codex_container
from codex_container import CodexContainer

HOST = podman_host.RemoteHost("fairy@box", identity="/id")


class ExecArgvTests(unittest.TestCase):
    def _container(self):
        c = CodexContainer(image="localhost/fairy-codex:latest", host=HOST)
        c.handle = podman_host.ContainerHandle(
            container_id="abc123", image="img", network=None, host=HOST)
        return c

    def test_exec_argv_wraps_podman_exec_over_ssh(self) -> None:
        argv = self._container().exec_argv(["codex", "--version"])
        self.assertEqual("ssh", argv[0])
        self.assertIn("fairy@box", argv)
        self.assertIn("-i", argv)  # ssh identity flag
        self.assertIn("/id", argv)
        # RemoteHost shlex-joins the remote command into the final arg.
        self.assertEqual("podman exec abc123 codex --version", argv[-1])

    def test_interactive_and_env_flags(self) -> None:
        argv = self._container().exec_argv(
            ["codex", "exec", "-"], interactive=True,
            env={"CODEX_HOME": codex_container.CONTAINER_CODEX_HOME},
        )
        self.assertEqual(
            "podman exec -i -e CODEX_HOME=/work/.codex-home abc123 "
            "codex exec -",
            argv[-1],
        )

    def test_read_file_cats_container_path(self) -> None:
        argv = self._container().exec_argv(["cat", "/work/.codex-run/out.json"])
        self.assertEqual(
            "podman exec abc123 cat /work/.codex-run/out.json", argv[-1])

    def test_exec_before_start_is_error(self) -> None:
        c = CodexContainer(image="img", host=HOST)
        with self.assertRaises(RuntimeError):
            c.exec_argv(["codex"])


if __name__ == "__main__":
    unittest.main()
