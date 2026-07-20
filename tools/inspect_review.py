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

Inspect saved OpenAI reviewer responses for analysis.

Reads the ``resp_*.json`` dumps written by
``pr_review_wrapper.py --debug-response-dir`` and surfaces the
parts that matter when comparing runs: the reasoning summaries (what
the model was thinking), the code_interpreter tool calls (what it
actually inspected in the repo), and the final review message.

Pass any run dir, ``openaidebug`` dir, or single JSON; files are found
recursively. ``--grep`` filters to responses whose reasoning, tool
calls, or message match the regex (case-insensitive).

Examples::

    tools/inspect_review.py simpast-runs/stock_a
    tools/inspect_review.py simpast-runs/bisect --pr 22624 --grep handler_name --show reasoning
    tools/inspect_review.py simpast-runs/bisect --grep handler_name --show all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def find_json(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("resp_*.json"))


def parse_one(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    resp = data.get("response") or {}
    out = resp.get("output") or []
    reasoning: list[str] = [
        s.get("text", "")
        for it in out if it.get("type") == "reasoning"
        for s in (it.get("summary") or [])
        if s.get("text")
    ]
    tools: list[str] = []
    for it in out:
        if it.get("type") != "code_interpreter_call":
            continue
        block = it.get("code") or ""
        for o in it.get("outputs") or []:
            if isinstance(o, dict):
                block += "\n" + (o.get("logs") or o.get("text") or "")
        tools.append(block)
    message = ""
    parsed: dict = {}
    for it in out:
        if it.get("type") == "message":
            for c in it.get("content") or []:
                if c.get("type") == "output_text":
                    message = c.get("text") or ""
    try:
        parsed = json.loads(message)
    except Exception:
        pass
    pr = ((data.get("wrapper_request") or {}).get("pull_request") or {}).get("number")
    return {
        "path": path,
        "pr": pr,
        "model": resp.get("model"),
        "is_main": bool(tools),
        "classification": parsed.get("classification") or parsed.get("route"),
        "reasoning": reasoning,
        "tools": tools,
        "message": parsed.get("message", message),
    }


def matches(rec: dict, pat: re.Pattern) -> bool:
    return bool(
        pat.search("\n".join(rec["reasoning"]))
        or pat.search("\n".join(rec["tools"]))
        or pat.search(rec["message"] or "")
    )


def title(text: str) -> str:
    return text.split("\n", 1)[0].strip("* ").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="run dir, openaidebug dir, or resp_*.json")
    ap.add_argument("--pr", type=int, help="only this PR number")
    ap.add_argument("--grep", help="filter to responses matching this regex (case-insensitive)")
    ap.add_argument("--show", choices=["compact", "reasoning", "tools", "message", "all"],
                    default="compact")
    ap.add_argument("--main-only", action="store_true",
                    help="skip triage (mini) calls; keep only the reviewer pass")
    args = ap.parse_args()

    pat = re.compile(args.grep, re.I) if args.grep else None
    for path in find_json(args.path):
        rec = parse_one(path)
        if not rec:
            continue
        if args.pr is not None and rec["pr"] != args.pr:
            continue
        if args.main_only and not rec["is_main"]:
            continue
        if pat and not matches(rec, pat):
            continue

        kind = "main" if rec["is_main"] else "triage"
        print(f"\n{'='*78}\n{path.parent.parent.name}/{path.name}  "
              f"PR#{rec['pr']} {kind} {rec['model']} class={rec['classification']} "
              f"reasoning_blocks={len(rec['reasoning'])} tool_calls={len(rec['tools'])} "
              f"msg_chars={len(rec['message'] or '')}")
        if args.show in ("compact", "all", "reasoning"):
            print("  reasoning: " + " | ".join(title(t) for t in rec["reasoning"]))
        if args.show in ("reasoning", "all"):
            for t in rec["reasoning"]:
                print("    - " + re.sub(r"\s+", " ", t).strip())
        if args.show in ("tools", "all"):
            for i, t in enumerate(rec["tools"]):
                print(f"    [tool {i}] " + re.sub(r"\s+", " ", t).strip()[:600])
        if args.show in ("message", "all"):
            print("  --- message ---\n" + (rec["message"] or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
