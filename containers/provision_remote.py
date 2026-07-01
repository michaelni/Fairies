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

Idempotent one-shot provisioning of a remote podman host for reviews.

Given a passwordless ssh destination (``user@host``) it:

1. checks the remote podman is reachable over ssh (``ssh DEST podman
   info``), with an actionable hint if the prerequisite is missing;
2. builds the review image on the remote, rebuilding only when the
   Containerfile changed (tracked via an image label);
3. seeds a bare mirror per repo so per-review provisioning only has to
   push the PR delta (see podman_repos).

Re-running is safe: existing image/mirrors are reused.

Prerequisite handled by root before this runs: install podman + the dev
packages on the host. Nothing else -- there is NO ``podman.socket`` and
NO lingering to enable, because the bot talks to podman purely as
``ssh DEST podman ...`` rather than ``podman --remote``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import podman_repos  # noqa: E402
from common import add_color_arg, setup_logging  # noqa: E402
from podman_host import (  # noqa: E402
    RemoteHost,
    build_image_if_needed,
    image_label,
)

logger = logging.getLogger("containers.provision_remote")

DEFAULT_TAG = "localhost/fairy-review:latest"
CONTEXT_DIR = REPO_ROOT / "containers"
DOCKERFILE = CONTEXT_DIR / "Containerfile"
CONTAINERFILE_LABEL = "fairy.containerfile-sha256"


def _run(argv: list[str], *, timeout_s: float = 60.0) -> subprocess.CompletedProcess:
    logger.info("$ %s", shlex.join(argv))
    cp = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout_s)
    if cp.stdout.strip():
        logger.debug("stdout: %s", cp.stdout.strip())
    if cp.stderr.strip():
        logger.debug("stderr: %s", cp.stderr.strip())
    return cp


def check_reachable(host: RemoteHost) -> None:
    cp = _run(host.argv(["podman", "info", "--format", "{{.Host.Arch}}"]))
    if cp.returncode != 0:
        raise RuntimeError(
            f"remote podman not reachable for {host.ssh_dest} ({cp.stderr.strip()}). "
            "Confirm the prerequisite: podman installed on the host and "
            "passwordless ssh works (try: ssh <dest> podman info)."
        )
    logger.info("remote podman reachable arch=%s", cp.stdout.strip())


def containerfile_sha256(dockerfile: Path) -> str:
    return hashlib.sha256(dockerfile.read_bytes()).hexdigest()


def ensure_image(
    *,
    host: RemoteHost,
    tag: str,
    dockerfile: Path,
    context_dir: Path,
    rebuild: bool,
) -> None:
    """Build the image on the remote, skipping if it already matches the
    current Containerfile (tracked via an image label)."""
    want = containerfile_sha256(dockerfile)
    have = image_label(tag, CONTAINERFILE_LABEL, host=host)
    stale = have != want
    if stale and have is not None:
        logger.info("image %s is stale (label=%s want=%s); rebuilding", tag, have, want)
    build_image_if_needed(
        image_tag=tag,
        dockerfile=dockerfile,
        context_dir=context_dir,
        force=rebuild or stale,
        labels={CONTAINERFILE_LABEL: want},
        host=host,
        build_timeout_s=7200.0,
    )
    logger.info("image ready tag=%s", tag)


def seed_mirrors(repo_roots: list[Path], host: RemoteHost, mirror_root: str) -> None:
    specs = podman_repos.build_repo_specs(repo_roots, mirror_root=mirror_root)
    for spec in specs:
        podman_repos.ensure_remote_mirror(host, spec.mirror_path)
        podman_repos.sync_repo_to_mirror(spec, host)
        logger.info("mirror seeded name=%s -> %s", spec.name, spec.mirror_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    p.add_argument("--ssh", required=True, metavar="USER@HOST", help="passwordless ssh destination")
    p.add_argument(
        "--identity", default=None,
        help="ssh identity (private key) file; optional, OpenSSH picks a "
             "default key / agent otherwise",
    )
    p.add_argument("--tag", default=DEFAULT_TAG, help="image tag (default: %(default)s)")
    p.add_argument("--context", type=Path, default=CONTEXT_DIR, help="build context")
    p.add_argument("--file", type=Path, default=DOCKERFILE, help="path to Containerfile")
    p.add_argument(
        "--mirror-root", default=podman_repos.DEFAULT_MIRROR_ROOT,
        help="remote bare-mirror root, relative to the ssh home (default: %(default)s)",
    )
    p.add_argument("--rebuild", action="store_true", help="force image rebuild even if current")
    p.add_argument(
        "repo_roots", nargs="*", type=Path,
        help="working-tree repos to seed as remote mirrors (optional)",
    )
    p.add_argument("--verbose", action="store_true", help="enable debug logging")
    add_color_arg(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(logger, args.verbose, color=args.color)

    host = RemoteHost(args.ssh, identity=args.identity)
    dockerfile = args.file if args.file.is_absolute() else (REPO_ROOT / args.file).resolve()

    check_reachable(host)
    ensure_image(
        host=host, tag=args.tag, dockerfile=dockerfile,
        context_dir=args.context.resolve(), rebuild=args.rebuild,
    )
    if args.repo_roots:
        seed_mirrors(args.repo_roots, host, args.mirror_root)
    else:
        logger.info("no repo_roots given; skipping mirror seeding")
    logger.info("provisioning complete host=%s", args.ssh)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(1) from exc
