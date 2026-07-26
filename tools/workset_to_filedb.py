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
 */

Mirror an old-layout workset repo dir into a filedb root (read-only on
the source). The old numeric ``state`` field becomes the directory and
is stripped from the content; everything else is copied verbatim.
Inspection aid for the filedb migration; with ``--move`` it becomes the
one-shot migration (run it with all fairy processes stopped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import filedb  # noqa: E402

# Old workset.WorkState value -> filedb directory. In-flight states
# (QUEUED/TRIAGE/REVIEW/COMBINE) become requests/: the old files carry
# no prepared payload (it lived in the pipeline's memory), so a queued/
# ticket would only crash the worker -- a rerun request makes the agent
# rebuild a fresh one.
STATE_DIRS = {
    1: "requests", 2: "requests", 3: "requests", 4: "requests",
    5: "reviewed", 6: "posted", 7: "skipped", 8: "cancelled", 9: "error",
}
# classification -> post action, as fairy.decision_from_review and
# issue_fairy.issue_review_decision derive it
ACTIONS = {"approve": "approve", "minor_issues_approve": "approve",
           "moderate_issues": "comment", "reply_no_verdict": "comment",
           "reply": "comment", "major_issues": "request_changes"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[-6])
    p.add_argument("source", type=Path, help="old-layout repo dir (pr-N.json files)")
    p.add_argument("target", type=Path, help="filedb root to create/fill")
    p.add_argument("--move", action="store_true",
                   help="delete source files after mirroring (migration mode)")
    args = p.parse_args()

    db = filedb.Db(args.target)
    n = 0
    for path in sorted(args.source.glob("*.json")):
        kind, _, num = path.stem.partition("-")
        if kind not in filedb.KINDS or not num.isdigit():
            print(f"skipping {path.name}: not a work item", file=sys.stderr)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        state = STATE_DIRS.get(data.pop("state", None))
        if state is None:
            print(f"skipping {path.name}: unknown state", file=sys.stderr)
            continue
        review = data.get("review") or {}
        # The old pipeline parked bare skip verdicts in REVIEWED (they
        # carried the backoff memory there); the filedb keeps only
        # operator-actionable verdicts in reviewed/, backoff lives in
        # skipped/.
        if state == "reviewed" and review.get("classification") == "skip" \
                and not review.get("label_changes"):
            state = "skipped"
        # The old skip counter n meant a 24*2^(n-1)h window. The filedb
        # stores the PRIOR backoff, and the agent's next wait is
        # max(24, 2*B): B must be HALF the old window or every migrated
        # ticket waits twice as long as before (found by the cutover
        # A/B: 8 parked items the old pipeline was already re-reviewing).
        skips = data.pop("consecutive_skip_count", 0) or 0
        if state == "skipped" and skips:
            data["skip_backoff_h"] = 12.0 * (2 ** (skips - 1))
        if state == "requests":
            data = {"action": "rerun"}
        # log_summary and the TUI's awaiting-you stats read the action
        # the new worker records; derive it for migrated verdicts
        if state == "reviewed" and "action" not in data \
                and review.get("classification") in ACTIONS:
            data["action"] = ACTIONS[review["classification"]]
        db.push(state, kind, int(num), data)
        if args.move:
            path.unlink()
        n += 1
    print(f"mirrored {n} items into {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
