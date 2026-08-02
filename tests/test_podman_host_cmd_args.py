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

Tests for fairy.podman_host_cmd_args.

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

    def test_forwards_codex_host_and_home(self) -> None:
        args = SimpleNamespace(
            podman_host=["fairy@x86box"],
            codex_host="fairy@codexbox", codex_home="/srv/fairy/codex")
        self.assertEqual(
            ["--podman", "--shell-host=fairy@x86box",
             "--codex-host=fairy@codexbox", "--codex-home=/srv/fairy/codex"],
            fairy.podman_host_cmd_args(args),
        )

    def test_codex_host_without_podman_host(self) -> None:
        args = SimpleNamespace(podman_host=[], codex_host="fairy@codexbox")
        self.assertEqual(
            ["--codex-host=fairy@codexbox"],
            fairy.podman_host_cmd_args(args),
        )

    def test_no_codex_flags_when_unset(self) -> None:
        args = SimpleNamespace(
            podman_host=["fairy@x86box"], codex_host=None, codex_home=None)
        self.assertEqual(
            ["--podman", "--shell-host=fairy@x86box"],
            fairy.podman_host_cmd_args(args),
        )


if __name__ == "__main__":
    unittest.main()
