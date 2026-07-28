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

Re-record the GitHub fixtures under tests/fixtures/github/.

The fixtures are captures of public GitHub API responses; ``MANIFEST``
below is the record of which request produced each one, so a fixture can
be refreshed instead of hand-edited when it drifts. Read-only: every path
here is a GET on a public repository, and no token is required (GitHub
allows 60 unauthenticated requests an hour, which is more than the
handful used here).

    tools/capture_github_fixtures.py --check     # compare shapes, write nothing
    tools/capture_github_fixtures.py --write     # re-record

``--check`` compares the key sets rather than the bytes: timestamps,
counters and avatar URLs churn constantly, while a renamed or dropped key
is what would actually break the adapters.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import add_color_arg, setup_logging  # noqa: E402

__all__ = ["MANIFEST", "capture", "shape_of"]

logger = logging.getLogger("capture_github_fixtures")

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "github"

_PR4_HEAD = "e3a94885362c5dd1d872073fa476e6c0bc6c7506"
_MID_RUN_COMMIT = "6014cfbfbf4474d3a9855d1ae8e4a179ee10a76f"

# Fixture -> the request that produced it. Every one comes from the
# project's own scratch repository, whose .github/workflows/fixtures.yml
# exists to emit each check-run conclusion on demand: a job that passes,
# one gated off, one that fails, and a slow one that can be caught
# running or cancelled by the next push.
MANIFEST: dict[str, str] = {
    "testrepo_pr4_statuses.json":
        f"/repos/michaelni/testrepo/commits/{_PR4_HEAD}/statuses",
    "testrepo_pr4_check_runs.json":
        f"/repos/michaelni/testrepo/commits/{_PR4_HEAD}/check-runs",
    "testrepo_check_runs_running.json":
        f"/repos/michaelni/testrepo/commits/{_MID_RUN_COMMIT}/check-runs",
    "testrepo_check_runs_cancelled.json":
        f"/repos/michaelni/testrepo/commits/{_MID_RUN_COMMIT}/check-runs",
    "testrepo_pr4_timeline.json":
        "/repos/michaelni/testrepo/issues/4/timeline?per_page=100",
    "testrepo_pr4_reviews.json": "/repos/michaelni/testrepo/pulls/4/reviews",
    "testrepo_pr4_review_comments.json":
        "/repos/michaelni/testrepo/pulls/4/comments",
    "testrepo_pr2_forcepush_timeline.json":
        "/repos/michaelni/testrepo/issues/2/timeline?per_page=100",
}


def capture(path: str) -> object:
    cmd = ["gcli", "-t", "github", "api", path]
    logger.info("+ %s", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"{path}: gcli exit {cp.returncode}: {cp.stderr.strip()}")
    return json.loads(cp.stdout)


def shape_of(obj: object) -> object:
    """The key structure of a payload, with the churning values dropped."""
    if isinstance(obj, dict):
        return {k: shape_of(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return shape_of(obj[0]) if obj else None
    return type(obj).__name__


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-record the GitHub fixtures under tests/fixtures/github/.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="report fixtures whose shape has drifted")
    mode.add_argument("--write", action="store_true", help="re-record")
    p.add_argument("-v", "--verbose", action="count", default=0)
    add_color_arg(p)
    args = p.parse_args()
    setup_logging(logger, bool(args.verbose), color=args.color)

    drifted = []
    for name, path in MANIFEST.items():
        live = capture(path)
        target = FIXTURES / name
        if args.write:
            target.write_text(json.dumps(live, indent=2) + "\n", encoding="utf-8")
            logger.info("wrote %s", target.relative_to(REPO_ROOT))
            continue
        stored = json.loads(target.read_text(encoding="utf-8"))
        if shape_of(stored) != shape_of(live):
            drifted.append(name)
            logger.warning("shape drift: %s", name)

    if drifted:
        logger.error("%d fixture(s) drifted; re-record with --write and "
                     "check the adapters still hold", len(drifted))
        return 1
    logger.info("%d fixture(s) %s", len(MANIFEST),
                "written" if args.write else "match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
