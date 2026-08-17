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

One-shot host setup that blocks a review container's egress to the LAN
(RFC1918 / link-local / CGNAT / private IPv6) and to the host itself
while leaving public-internet egress intact -- internet access is
useful during review. The filter lives on the HOST, outside the
container, so a fully compromised container cannot remove it. Default mode is dry-run (every
command is printed, nothing runs); ``--apply`` executes.

For rootless podman (``--rootless-user USER``, the production model): the
network backend (slirp4netns or pasta) forwards container egress from a
host socket owned by USER's uid, so the block is an OUTPUT drop keyed on
``meta skuid <uid>`` -- the container cannot change that uid or touch
host nftables, so it is bypass-proof from inside. Host-local
destinations (``fib daddr type local``: loopback and every address the
host owns, including its public one, which no LAN range covers) are
dropped too, so the account cannot reach host-only services such as
sshd on 127.0.0.1. DNS (port 53) is exempted so name resolution still
works when the host resolver is a LAN or loopback address
(e.g. systemd-resolved on 127.0.0.53). USER must be an account
dedicated to running review containers -- every LAN connection it opens
is dropped, so it must not also run the fairy orchestrator or anything
else that needs the LAN. ``--apply``
loads the ruleset now and installs a systemd unit that reloads it on
boot; run it as root.

The subnet/FORWARD path (no ``--rootless-user``) is the pre-existing
rootful setup; see ``configure_host``.
"""

from __future__ import annotations

import argparse
import logging
import os
import pwd
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

DEFAULT_ROOTLESS_TABLE = "fairy_egress"
DEFAULT_NFT_FILE = Path("/etc/fairy-egress.nft")
DEFAULT_UNIT_FILE = Path("/etc/systemd/system/fairy-egress.service")

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

# A LAN host reachable only via a global IPv6 (GUA) is not covered: no
# reserved prefix separates it from the public internet.
LAN_BLOCK_DESTS6 = ("fc00::/7", "fe80::/10")


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


def build_rootless_egress_nft(*, uid: int, table: str) -> str:
    """Render the idempotent uid-keyed OUTPUT ruleset (see module docstring).

    ``add``+``flush`` make re-running converge to the same state. Traffic
    from ``uid`` to the LAN ranges and to host-local addresses is
    dropped; DNS (port 53) and everything to the public internet pass.
    """
    dests = ", ".join(LAN_BLOCK_DESTS)
    dests6 = ", ".join(LAN_BLOCK_DESTS6)
    return (
        f"add table inet {table}\n"
        f"flush table inet {table}\n"
        f"add chain inet {table} output {{ type filter hook output priority filter - 10; policy accept; }}\n"
        f"add rule inet {table} output meta skuid {uid} udp dport 53 accept\n"
        f"add rule inet {table} output meta skuid {uid} tcp dport 53 accept\n"
        f"# Restrict DNS to one resolver by dropping the two rules above and adding e.g.:\n"
        f"# add rule inet {table} output meta skuid {uid} ip daddr 192.0.2.53 udp dport 53 accept\n"
        f"add rule inet {table} output meta skuid {uid} ip daddr {{ {dests} }} drop\n"
        f"add rule inet {table} output meta skuid {uid} ip6 daddr {{ {dests6} }} drop\n"
        f"add rule inet {table} output meta skuid {uid} fib daddr type local drop\n"
    )


def build_egress_unit(*, nft_file: Path) -> str:
    """Render the systemd oneshot that reloads the ruleset on boot."""
    return (
        "[Unit]\n"
        "Description=fairy rootless podman LAN-egress block\n"
        "After=nftables.service network-pre.target\n"
        "Wants=network-pre.target\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        f"ExecStart=nft -f {nft_file}\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def configure_rootless_egress(
    *, user: str, table: str, nft_file: Path, unit_file: Path, apply: bool,
) -> int:
    """Print or install the rootless uid-keyed LAN-egress block.

    ``--apply`` writes ``nft_file`` and ``unit_file``, loads the ruleset
    now with ``nft -f``, and enables the unit so it reloads on boot; it
    must run as root. The dry-run prints exactly these steps; run as
    root it also validates the script with ``nft --check`` (unprivileged
    nft cannot open the netlink socket even to check).

    Loading is an explicit ``nft -f`` rather than starting the unit: the
    unit is ``RemainAfterExit`` so ``systemctl start`` is a no-op once it
    is active, which would leave an edited ruleset written-but-not-loaded
    on a re-apply. Returns a process exit code.
    """
    try:
        uid = pwd.getpwnam(user).pw_uid
    except KeyError:
        logger.error("no such user %r on this host", user)
        return 2
    if uid == 0:
        logger.error("refusing to block root's egress (uid 0)")
        return 2
    nft_script = build_rootless_egress_nft(uid=uid, table=table)
    unit = build_egress_unit(nft_file=nft_file)
    commands = [
        ["nft", "-f", str(nft_file)],
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", unit_file.name],
    ]

    if not apply:
        logger.info("dry-run; pass --apply (as root) to install the following:")
        logger.info("write %s:\n%s", nft_file, nft_script)
        logger.info("write %s:\n%s", unit_file, unit)
        for cmd in commands:
            logger.info("$ %s", shlex.join(cmd))
        if os.geteuid() == 0:
            return _run(["nft", "--check", "-f", "-"], input_text=nft_script)
        return 0

    if os.geteuid() != 0:
        logger.error(
            "--apply for rootless egress must run as root "
            "(it writes %s and installs a systemd unit)", nft_file,
        )
        return 1

    logger.info("blocking LAN egress for user %s (uid %d)", user, uid)
    nft_file.write_text(nft_script, encoding="utf-8")
    unit_file.write_text(unit, encoding="utf-8")
    for cmd in commands:
        rc = _run(cmd)
        if rc != 0:
            return rc
    return 0


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
        if os.geteuid() == 0:
            return _run(["nft", "--check", "-f", "-"], input_text=nft_script)
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
    p.add_argument(
        "--rootless-user",
        help="rootless podman account whose LAN egress to block via a "
             "uid-keyed OUTPUT rule (the production model; run --apply as root)",
    )
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
    if args.rootless_user:
        return configure_rootless_egress(
            user=args.rootless_user, table=DEFAULT_ROOTLESS_TABLE,
            nft_file=DEFAULT_NFT_FILE, unit_file=DEFAULT_UNIT_FILE, apply=args.apply,
        )
    return configure_host(
        network=args.network_name,
        subnet=args.subnet,
        table=args.table_name,
        apply=args.apply,
        use_sudo=args.sudo,
    )


if __name__ == "__main__":
    raise SystemExit(main())
