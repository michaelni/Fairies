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
import tempfile
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


ROOTLESS_UID = 4242
ROOTLESS_TABLE = "fairy_egress_test"


def _pwd_entry(uid: int) -> mock.Mock:
    entry = mock.Mock()
    entry.pw_uid = uid
    return entry


class BuildRootlessEgressNftTests(unittest.TestCase):
    def test_keys_on_uid_exempts_dns_and_drops_lan(self) -> None:
        script = setup_host.build_rootless_egress_nft(
            uid=ROOTLESS_UID, table=ROOTLESS_TABLE)
        self.assertIn(f"add table inet {ROOTLESS_TABLE}", script)
        self.assertIn("hook output priority filter - 10", script)
        # DNS exempted before the LAN drop, so name resolution via a LAN
        # resolver survives.
        for proto in ("udp", "tcp"):
            self.assertIn(
                f"meta skuid {ROOTLESS_UID} {proto} dport 53 accept", script)
        for dest in setup_host.LAN_BLOCK_DESTS:
            self.assertIn(dest, script)
        for dest in setup_host.LAN_BLOCK_DESTS6:
            self.assertIn(dest, script)
        self.assertIn(f"meta skuid {ROOTLESS_UID} ip daddr", script)
        self.assertIn(
            f"meta skuid {ROOTLESS_UID} fib daddr type local drop", script)
        self.assertLess(script.index("dport 53 accept"),
                        script.index("ip daddr"))
        self.assertLess(script.index("dport 53 accept"),
                        script.index("fib daddr type local"))


class ConfigureRootlessEgressTests(unittest.TestCase):
    def _kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = dict(
            user="sandbox", table=ROOTLESS_TABLE,
            nft_file=Path("/etc/fairy-egress.nft"),
            unit_file=Path("/etc/systemd/system/fairy-egress.service"),
            apply=False,
        )
        base.update(overrides)
        return base

    def test_dry_run_writes_nothing_and_runs_nothing(self) -> None:
        with mock.patch.object(setup_host.pwd, "getpwnam",
                               return_value=_pwd_entry(ROOTLESS_UID)), \
             mock.patch.object(setup_host.subprocess, "run") as run, \
             mock.patch.object(Path, "write_text") as write:
            rc = setup_host.configure_rootless_egress(**self._kwargs())
        self.assertEqual(0, rc)
        self.assertFalse(run.called)
        self.assertFalse(write.called)

    def test_unknown_user_fails(self) -> None:
        with mock.patch.object(setup_host.pwd, "getpwnam",
                               side_effect=KeyError("sandbox")):
            rc = setup_host.configure_rootless_egress(**self._kwargs(apply=True))
        self.assertEqual(2, rc)

    def test_refuses_root_uid(self) -> None:
        with mock.patch.object(setup_host.pwd, "getpwnam",
                               return_value=_pwd_entry(0)):
            rc = setup_host.configure_rootless_egress(**self._kwargs(apply=True))
        self.assertEqual(2, rc)

    def test_apply_as_non_root_refuses(self) -> None:
        with mock.patch.object(setup_host.pwd, "getpwnam",
                               return_value=_pwd_entry(ROOTLESS_UID)), \
             mock.patch.object(setup_host.os, "geteuid", return_value=1000), \
             mock.patch.object(setup_host.subprocess, "run") as run, \
             mock.patch.object(Path, "write_text") as write:
            rc = setup_host.configure_rootless_egress(**self._kwargs(apply=True))
        self.assertEqual(1, rc)
        self.assertFalse(run.called)
        self.assertFalse(write.called)

    def test_apply_as_root_writes_files_and_enables_unit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            nft_file = Path(d) / "fairy-egress.nft"
            unit_file = Path(d) / "fairy-egress.service"
            with mock.patch.object(setup_host.pwd, "getpwnam",
                                   return_value=_pwd_entry(ROOTLESS_UID)), \
                 mock.patch.object(setup_host.os, "geteuid", return_value=0), \
                 mock.patch.object(setup_host.subprocess, "run",
                                   return_value=_completed(0)) as run:
                rc = setup_host.configure_rootless_egress(
                    **self._kwargs(apply=True, nft_file=nft_file, unit_file=unit_file))
            self.assertEqual(0, rc)
            self.assertIn(f"meta skuid {ROOTLESS_UID}",
                          nft_file.read_text(encoding="utf-8"))
            self.assertIn(str(nft_file), unit_file.read_text(encoding="utf-8"))
        commands = [call.args[0] for call in run.call_args_list]
        # nft -f loads now; enable (not --now) only persists across boot,
        # since the RemainAfterExit unit would no-op a re-apply's start.
        self.assertEqual(["nft", "-f", str(nft_file)], commands[0])
        self.assertIn(["systemctl", "daemon-reload"], commands)
        self.assertIn(["systemctl", "enable", unit_file.name], commands)
        self.assertNotIn(
            ["systemctl", "enable", "--now", unit_file.name], commands)


if __name__ == "__main__":
    unittest.main()
