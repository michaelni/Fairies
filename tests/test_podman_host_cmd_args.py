"""Tests for fairy.podman_host_cmd_args.

Locks down the thin flag-injection contract: each --podman-host value
turns into a wrapper --shell-host flag (plus --podman), and the result
is empty when unset.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


class PodmanHostCmdArgsTests(unittest.TestCase):
    def test_empty_when_podman_host_unset(self) -> None:
        args = SimpleNamespace(podman_host=[])
        self.assertEqual([], fairy.podman_host_cmd_args(args))

    def test_injects_container_flags(self) -> None:
        args = SimpleNamespace(podman_host=["fairy@podman-host"])
        self.assertEqual(
            ["--podman", "--shell-host=fairy@podman-host"],
            fairy.podman_host_cmd_args(args),
        )

    def test_forwards_each_host_in_order(self) -> None:
        args = SimpleNamespace(
            podman_host=["fairy@x86box", "arm64=fairy@ampere,cpus=12"])
        self.assertEqual(
            ["--podman", "--shell-host=fairy@x86box",
             "--shell-host=arm64=fairy@ampere,cpus=12"],
            fairy.podman_host_cmd_args(args),
        )


if __name__ == "__main__":
    unittest.main()
