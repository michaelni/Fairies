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

Print the per-PR review verdict + body from simpast cell run.log(s).

The bot logs a final per-PR summary block: a ``PR #<n>: <verdict>``
header line followed by the posted message body. Parallel review
subprocesses interleave ``[wrapper ...]`` debug lines into the same log,
and the bot prefixes its own lines with a ``<ts> <level> `` stamp. Both
are stripped so the output is just the reviews.

Usage:
    tools/extract_suite_reviews.py <cell-dir-or-run.log> [PR]
"""
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [A-Z] ?")
HEADER = re.compile(r"^PR #(\d+):\s")
STOP = re.compile(
    r"^(gcli_cache |bot_state |Summary:|Auto-merge|Scheduled|Would auto"
    r"|LLM review classifications|Dry-run|PR #)"
)


def extract(run_log: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for raw in ANSI.sub("", run_log.read_text(errors="replace")).splitlines():
        if raw.startswith("[wrapper"):
            continue
        line = STAMP.sub("", raw)
        if HEADER.match(line):
            cur = out.setdefault(HEADER.match(line).group(1), [])
            cur.append(line)
        elif cur is not None:
            if STOP.match(line):
                cur = None
            else:
                cur.append(line)
    return out


if __name__ == "__main__":
    arg = Path(sys.argv[1])
    run_log = arg if arg.is_file() else arg / "run.log"
    want = sys.argv[2] if len(sys.argv) > 2 else None
    reviews = extract(run_log)
    for pr in sorted(reviews):
        if want and pr != want:
            continue
        print(f"================= {run_log.parent.name}  PR #{pr} =================")
        print("\n".join(reviews[pr]).strip())
        print()
