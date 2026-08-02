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

Tests for ``containers/setup_host``.

These do NOT touch real podman or nftables; every external command
is mocked. The point is to lock down:
- the exact nftables script that gets shipped to ``nft``
- the podman network argv (subnet must be passed)
- dry-run vs --apply behavior (no subprocess calls in dry-run)
- the sudo prefix gating
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONTAINERS_DIR = REPO_ROOT / "containers"
if str(CONTAINERS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTAINERS_DIR))

import setup_host  # noqa: E402


# RFC 5737 TEST-NET-1; intentionally unrelated to ``setup_host.DEFAULT_SUBNET``
# so these tests verify parameter pass-through rather than the production
# default value.
TEST_SUBNET = "192.0.2.0/24"
TEST_NETWORK = "fairy-test-net"
TEST_TABLE = "fairy_test_table"


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> mock.Mock:
    cp = mock.Mock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _cfg_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        network=TEST_NETWORK, subnet=TEST_SUBNET, table=TEST_TABLE,
        apply=False, use_sudo=False,
    )
    base.update(overrides)
    return base


class BuildNftScriptTests(unittest.TestCase):
    def test_includes_subnet_table_and_drop_destinations(self) -> None:
        script = setup_host.build_nft_script(subnet=TEST_SUBNET, table=TEST_TABLE)
        self.assertIn(f"add table inet {TEST_TABLE}", script)
        self.assertIn(f"flush table inet {TEST_TABLE}", script)
        self.assertIn(
            f"ip saddr {TEST_SUBNET} ip daddr {TEST_SUBNET} return",
            script,
        )
        for dest in setup_host.LAN_BLOCK_DESTS:
            self.assertIn(dest, script)

    def test_uses_priority_lower_than_default_filter(self) -> None:
        # `priority filter - 10` runs before podman's default-priority
        # rules; if a future edit silently changes this we want to
        # notice rather than discover at runtime that drops never fire.
        script = setup_host.build_nft_script(subnet=TEST_SUBNET, table=TEST_TABLE)
        self.assertIn("priority filter - 10", script)


class BuildPodmanNetworkArgvTests(unittest.TestCase):
    def test_passes_fixed_subnet(self) -> None:
        argv = setup_host.build_podman_network_argv(name=TEST_NETWORK, subnet=TEST_SUBNET)
        self.assertEqual(
            ["podman", "network", "create", f"--subnet={TEST_SUBNET}", TEST_NETWORK],
            argv,
        )


class ConfigureHostTests(unittest.TestCase):
    def test_dry_run_does_not_invoke_subprocess(self) -> None:
        with mock.patch.object(setup_host.subprocess, "run") as run:
            rc = setup_host.configure_host(**_cfg_kwargs())
        self.assertEqual(0, rc)
        self.assertFalse(run.called)

    def test_apply_creates_network_then_loads_nft(self) -> None:
        # network missing -> create called -> nft loaded.
        results = [
            _completed(1),                     # network exists check
            _completed(0, stdout="netid"),     # network create
            _completed(0),                     # nft -f -
        ]
        with mock.patch.object(setup_host.subprocess, "run", side_effect=results) as run:
            rc = setup_host.configure_host(**_cfg_kwargs(apply=True))
        self.assertEqual(0, rc)
        self.assertEqual(3, run.call_count)
        self.assertEqual(
            ["podman", "network", "exists", TEST_NETWORK],
            run.call_args_list[0].args[0],
        )
        self.assertEqual(
            ["podman", "network", "create", f"--subnet={TEST_SUBNET}", TEST_NETWORK],
            run.call_args_list[1].args[0],
        )
        nft_call = run.call_args_list[2]
        self.assertEqual(["nft", "-f", "-"], nft_call.args[0])
        self.assertIn(f"add table inet {TEST_TABLE}", nft_call.kwargs["input"])

    def test_apply_skips_create_when_network_exists(self) -> None:
        results = [_completed(0), _completed(0)]
        with mock.patch.object(setup_host.subprocess, "run", side_effect=results) as run:
            rc = setup_host.configure_host(**_cfg_kwargs(apply=True))
        self.assertEqual(0, rc)
        self.assertEqual(2, run.call_count)
        # second call is nft, not podman create.
        self.assertEqual(["nft", "-f", "-"], run.call_args_list[1].args[0])

    def test_sudo_prefix_only_applied_to_nft(self) -> None:
        results = [_completed(0), _completed(0)]
        with mock.patch.object(setup_host.subprocess, "run", side_effect=results) as run:
            setup_host.configure_host(**_cfg_kwargs(apply=True, use_sudo=True))
        self.assertEqual(["sudo", "nft", "-f", "-"], run.call_args_list[1].args[0])

    def test_aborts_on_network_create_failure(self) -> None:
        results = [
            _completed(1),                          # exists check: missing
            _completed(125, stderr="boom"),         # create fails
        ]
        with mock.patch.object(setup_host.subprocess, "run", side_effect=results) as run:
            rc = setup_host.configure_host(**_cfg_kwargs(apply=True))
        self.assertEqual(125, rc)
        self.assertEqual(2, run.call_count)


if __name__ == "__main__":
    unittest.main()
