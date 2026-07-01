#!/usr/bin/env python3
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

One-shot host setup for the fairy isolated review network.

Two outputs:
1. A podman network with a fixed IPv4 subnet so the nftables rules
   can reference it.
2. An nftables ``inet`` table that drops egress from that subnet to
   RFC1918 / link-local / CGNAT, allowing everything else (the
   public internet).

Default mode is dry-run: every command is printed, nothing is run.
``--apply`` actually executes them. Per the rule that the network
filter must live OUTSIDE the container, the nftables rules are
installed on the host and a compromised container cannot remove them.

Scope: this ``ip saddr <subnet>`` FORWARD filter only works under
**rootful** podman, where netavark puts a real bridge on the host and
container egress traverses the host FORWARD chain with the container
subnet as source. It does NOT apply to rootless podman: both pasta
(the rootless default) and slirp4netns translate egress to the host's
own address in userspace, so the host FORWARD chain never sees the
container subnet as a source and these rules match nothing.

TODO(egress-isolation, rootless): the production deployment runs
rootless podman in a dedicated account, so this script is not wired
into it yet and review containers currently have unrestricted egress
(internet AND LAN). The rootless-compatible block is a root-installed
nftables rule keyed on the fairy uid -- pasta opens its outbound
sockets as that uid, so e.g. ``meta skuid <uid> ct state new ip daddr
{ RFC1918... } drop`` in OUTPUT is bypass-proof (the container cannot
change its uid or touch host nftables). Confirm pasta carries skuid on
the target box before relying on it, then add it as root pre-setup.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import add_color_arg, setup_logging  # noqa: E402

logger = logging.getLogger("containers.setup_host")

DEFAULT_NETWORK_NAME = "fairy-isolated"
DEFAULT_SUBNET = "10.222.0.0/24"
DEFAULT_TABLE_NAME = "fairy_isolation"

# Destinations the container must NOT be able to reach. Excludes the
# container's own subnet (added to the allow rule below) so intra-net
# traffic still works.
LAN_BLOCK_DESTS = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "100.64.0.0/10",
)


def build_nft_script(*, subnet: str, table: str) -> str:
    """Render an idempotent nftables ruleset.

    ``add table`` is a no-op if the table already exists; ``flush
    table`` then clears any prior rules so re-running this script
    produces the same final state regardless of starting state.
    """
    dests = ", ".join(LAN_BLOCK_DESTS)
    return (
        f"add table inet {table}\n"
        f"flush table inet {table}\n"
        f"add chain inet {table} forward {{ type filter hook forward priority filter - 10; policy accept; }}\n"
        f"add rule inet {table} forward ip saddr {subnet} ip daddr {subnet} return\n"
        f"add rule inet {table} forward ip saddr {subnet} ip daddr {{ {dests} }} drop\n"
    )


def build_podman_network_argv(*, name: str, subnet: str) -> list[str]:
    return ["podman", "network", "create", f"--subnet={subnet}", name]


def _run(argv: list[str], *, input_text: str | None = None) -> int:
    logger.info("$ %s", shlex.join(argv))
    if input_text is not None:
        logger.debug("stdin (%d bytes):\n%s", len(input_text), input_text.rstrip())
    try:
        cp = subprocess.run(
            argv, input=input_text, text=True,
            capture_output=True, check=False, timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.error("command timed out: %s", shlex.join(argv))
        return 124
    if cp.stdout:
        logger.debug("stdout: %s", cp.stdout.rstrip())
    if cp.stderr:
        logger.debug("stderr: %s", cp.stderr.rstrip())
    if cp.returncode != 0:
        logger.error("rc=%d: %s", cp.returncode, cp.stderr.rstrip() or cp.stdout.rstrip())
    return cp.returncode


def _podman_network_exists(name: str) -> bool:
    cp = subprocess.run(
        ["podman", "network", "exists", name],
        capture_output=True, check=False, timeout=30,
    )
    return cp.returncode == 0


def _maybe_sudo(argv: list[str], *, use_sudo: bool) -> list[str]:
    return ["sudo", *argv] if use_sudo else argv


def configure_host(
    *,
    network: str,
    subnet: str,
    table: str,
    apply: bool,
    use_sudo: bool,
) -> int:
    """Print or apply the host configuration. Returns a process exit code."""
    nft_script = build_nft_script(subnet=subnet, table=table)
    podman_create_argv = build_podman_network_argv(name=network, subnet=subnet)
    nft_argv = _maybe_sudo(["nft", "-f", "-"], use_sudo=use_sudo)

    if not apply:
        logger.info("dry-run; pass --apply to execute the following:")
        logger.info("$ %s", shlex.join(podman_create_argv))
        logger.info("$ %s <<EOF\n%sEOF", shlex.join(nft_argv), nft_script)
        return 0

    if _podman_network_exists(network):
        logger.info("podman network %s already exists; skipping create", network)
    else:
        rc = _run(podman_create_argv)
        if rc != 0:
            return rc

    return _run(nft_argv, input_text=nft_script)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    p.add_argument("--network-name", default=DEFAULT_NETWORK_NAME, help="podman network name")
    p.add_argument("--subnet", default=DEFAULT_SUBNET, help="IPv4 subnet for the isolated network")
    p.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="nftables table name")
    p.add_argument("--apply", action="store_true", help="actually run commands (default: print only)")
    p.add_argument(
        "--sudo", action="store_true",
        help="prefix nftables commands with sudo (only with --apply)",
    )
    p.add_argument("--verbose", action="store_true", help="enable debug logging")
    add_color_arg(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(logger, args.verbose, color=args.color)
    return configure_host(
        network=args.network_name,
        subnet=args.subnet,
        table=args.table_name,
        apply=args.apply,
        use_sudo=args.sudo,
    )


if __name__ == "__main__":
    raise SystemExit(main())
