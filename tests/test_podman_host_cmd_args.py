"""Tests for fairy.podman_host_cmd_args.

Locks down the thin flag-injection contract: --podman-host turns into
the wrapper's --podman / --podman-ssh-dest flags, and is empty when
unset.
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
        args = SimpleNamespace(podman_host=None)
        self.assertEqual([], fairy.podman_host_cmd_args(args))

    def test_injects_container_flags(self) -> None:
        args = SimpleNamespace(podman_host="fairy@192.168.2.4")
        self.assertEqual(
            ["--podman", "--podman-ssh-dest=fairy@192.168.2.4"],
            fairy.podman_host_cmd_args(args),
        )


if __name__ == "__main__":
    unittest.main()
