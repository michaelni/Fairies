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

Build the fairy review container image (shared by all wrappers).

Thin CLI around ``podman_host.build_image_if_needed`` so operators and
CI can refresh the image without touching any OpenAI-specific wrapper.
podman runs on the ssh host (``ssh DEST podman build``); the build
context is streamed there as a tar, so no local podman is needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import add_color_arg, setup_logging  # noqa: E402
from podman_host import RemoteHost, build_image_if_needed  # noqa: E402

logger = logging.getLogger("containers.build_image")

DEFAULT_TAG = "localhost/fairy-review:latest"
CONTEXT_DIR = REPO_ROOT / "containers"
DOCKERFILE = CONTEXT_DIR / "Containerfile"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    p.add_argument("--tag", default=DEFAULT_TAG, help="Image tag (default: %(default)s)")
    p.add_argument(
        "--context",
        type=Path,
        default=CONTEXT_DIR,
        help="Build context (default: repo containers/)",
    )
    p.add_argument(
        "--file",
        type=Path,
        default=DOCKERFILE,
        help="Path to Containerfile (default: containers/Containerfile)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the tag already exists on the host",
    )
    p.add_argument(
        "--ssh", required=True, metavar="USER@HOST",
        help="ssh destination of the podman host to build on",
    )
    p.add_argument(
        "--identity", default=None,
        help="ssh identity (private key) file; optional, OpenSSH picks a "
             "default key / agent otherwise",
    )
    p.add_argument("--port", type=int, default=None, help="ssh port of --ssh")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    add_color_arg(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(logger, args.verbose, color=args.color)
    ctx = args.context.resolve()
    dockerfile = args.file
    if not dockerfile.is_absolute():
        dockerfile = (REPO_ROOT / dockerfile).resolve()
    logger.info(
        "building image tag=%s dockerfile=%s context=%s force=%s host=%s",
        args.tag, dockerfile, ctx, args.force, args.ssh,
    )
    build_image_if_needed(
        image_tag=args.tag,
        dockerfile=dockerfile,
        context_dir=ctx,
        force=args.force,
        host=RemoteHost(args.ssh, identity=args.identity, port=args.port),
        build_timeout_s=7200.0,
    )
    logger.info("image ready tag=%s", args.tag)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(1) from exc
