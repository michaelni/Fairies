#!/usr/bin/env python3
"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

Diff the two cutover A/B arms (tools/ab_run.sh).

Arm A = old pipeline (master/production tree), arm B = the filedb
agent; both ran with tools/ab_stub_llm.py, so "this item reached the
LLM" is a payload file and the review inputs are byte-comparable.

Buckets:
  MATCH    -- reviewed by both arms with identical payloads (count only)
  DRIFT    -- the item changed on the forge between the two runs
              (pull_request/issue updated_at differs): excluded
  MISMATCH -- reviewed by only one arm, or payloads differ; each line
              shows the other arm's outcome (arm B from the filedb,
              arm A from its log) so the diverging gate is nameable

--allowlist FILE: one regex per line (# comments); a MISMATCH line any
regex matches is reported as allowed and does not fail the run.
Exit: 0 clean/allowed, 1 unallowed mismatches. Throwaway tooling.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import filedb  # noqa: E402


def load_payloads(d: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for p in d.glob("*.json"):
        kind, _, num = p.stem.partition("-")
        out[(kind, int(num))] = json.loads(p.read_text(encoding="utf-8"))
    return out


def updated_at(payload: dict) -> str | None:
    item = payload.get("pull_request") or payload.get("issue") or {}
    return item.get("updated_at")


def json_diff_paths(a, b, prefix="", out=None, cap=8) -> list[str]:
    """First differing paths between two json values, dotted."""
    if out is None:
        out = []
    if len(out) >= cap:
        return out
    if type(a) is not type(b):
        out.append(f"{prefix or '.'} (type {type(a).__name__} vs {type(b).__name__})")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{prefix}.{k} (only in {'B' if k not in a else 'A'})")
                if len(out) >= cap:
                    return out
            else:
                json_diff_paths(a[k], b[k], f"{prefix}.{k}", out, cap)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{prefix}[] (len {len(a)} vs {len(b)})")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                json_diff_paths(x, y, f"{prefix}[{i}]", out, cap)
    elif a != b:
        out.append(prefix or ".")
    return out


def db_outcome(db: filedb.Db, kind: str, number: int) -> str:
    state = db.find(kind, number)
    if state is None:
        return "no ticket"
    data = db.get(state, kind, number) or {}
    reason = (data.get("reason") or data.get("error")
              or (data.get("review") or {}).get("classification") or "")
    return f"{state}/ {reason}".strip()


def log_context(logs: list[Path], kind: str, number: int, cap=2) -> str:
    pat = re.compile(rf"#\s?{number}\b")
    hits = []
    for log in logs:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if pat.search(line) and kind[:2] in line.lower():
                hits.append(line.strip())
    return " | ".join(hits[-cap:]) if hits else "(no log line)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[10])
    p.add_argument("--payloads-a", type=Path, required=True)
    p.add_argument("--payloads-b", type=Path, required=True)
    p.add_argument("--new-db", type=Path, required=True)
    p.add_argument("--log-a", type=Path, action="append", default=[])
    p.add_argument("--allowlist", type=Path)
    args = p.parse_args()

    a = load_payloads(args.payloads_a)
    b = load_payloads(args.payloads_b)
    db = filedb.Db(args.new_db)
    allow = []
    if args.allowlist:
        for line in args.allowlist.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allow.append(re.compile(line))

    mismatches, drift, matched = [], [], 0
    for key in sorted(set(a) & set(b)):
        if updated_at(a[key]) != updated_at(b[key]):
            drift.append(f"{key[0]} #{key[1]}: updated_at moved between runs")
            continue
        paths = json_diff_paths(a[key], b[key])
        if paths:
            mismatches.append(
                f"{key[0]} #{key[1]}: payloads differ at {', '.join(paths)}")
        else:
            matched += 1
    for kind, num in sorted(set(a) - set(b)):
        mismatches.append(
            f"{kind} #{num}: only arm A reviewed it; arm B: "
            + db_outcome(db, kind, num))
    for kind, num in sorted(set(b) - set(a)):
        mismatches.append(
            f"{kind} #{num}: only arm B reviewed it; arm A: "
            + log_context(args.log_a, kind, num))

    print(f"MATCH    {matched} item(s) reviewed by both arms, payloads identical")
    for line in drift:
        print(f"DRIFT    {line}")
    failures = 0
    for line in mismatches:
        if any(rx.search(line) for rx in allow):
            print(f"ALLOWED  {line}")
        else:
            print(f"MISMATCH {line}")
            failures += 1
    print(f"\n{failures} unallowed mismatch(es), {len(drift)} drift, "
          f"{len(mismatches) - failures} allowed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
