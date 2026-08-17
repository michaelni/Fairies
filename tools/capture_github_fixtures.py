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

The fixtures are captures of GitHub API responses from the project's own
scratch repository; ``MANIFEST`` below is the record of which request
produced each one, so a fixture can be refreshed instead of hand-edited
when it drifts. Read-only: every path here is a GET on a public
repository, and no token is required.

Two entries share a commit on purpose. ``testrepo_check_runs_running``
was taken while the slow job was still going and ``_cancelled`` after
the next push cancelled it. A run cannot be caught mid-flight twice, so
that one is listed in ``POINT_IN_TIME``: ``--check`` reports it as
skipped rather than drifted, because the state it holds is gone and
comparing against a finished run would cry wolf on every call.

    tools/capture_github_fixtures.py --check     # compare shapes, write nothing
    tools/capture_github_fixtures.py --write     # re-record

``--check`` compares the key structure rather than the bytes: timestamps,
counters, avatar URLs and which optional fields happen to be null all
churn constantly, while a renamed or dropped key is what would actually
break the adapters.
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

__all__ = ["MANIFEST", "POINT_IN_TIME", "capture", "shape_of"]

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
    "testrepo_review_request_timeline.json":
        "/repos/michaelni/testrepo/issues/5/timeline?per_page=100",
}


# Fixtures holding a state that cannot be observed again. Kept in
# MANIFEST so their provenance is recorded and --write can still reach
# the endpoint, but excluded from the drift comparison.
POINT_IN_TIME = frozenset({"testrepo_check_runs_running.json"})


def capture(path: str) -> object:
    cmd = ["gcli", "-t", "github", "api", path]
    logger.info("+ %s", " ".join(cmd))
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"{path}: gcli exit {cp.returncode}: {cp.stderr.strip()}")
    return json.loads(cp.stdout)


def shape_of(obj: object) -> object:
    """The key structure of a payload, with the values dropped.

    Scalars all collapse to one marker rather than to their type name:
    GitHub nulls an optional field whenever the data has nothing to put
    there -- ``start_line`` on a single-line review comment, ``commit_id``
    on an entry that names no commit -- so comparing types reports drift
    every time the underlying rows differ. A renamed or dropped key still
    shows, which is what an adapter actually breaks on.
    """
    if isinstance(obj, dict):
        return {k: shape_of(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        # Merge every element: a timeline holds several entry kinds, and
        # reading only the first would miss a change in any of the rest.
        merged: dict = {}
        for item in obj:
            shape = shape_of(item)
            if isinstance(shape, dict):
                merged.update(shape)
            elif shape is not None:
                merged[shape] = shape
        return merged or None
    return "scalar"


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
        if args.check and name in POINT_IN_TIME:
            logger.info("skipped (point-in-time): %s", name)
            continue
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
    logger.info("%d of %d fixture(s) %s", len(MANIFEST) - (
                    len(POINT_IN_TIME) if args.check else 0), len(MANIFEST),
                "written" if args.write else "match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
