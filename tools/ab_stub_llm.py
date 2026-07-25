#!/usr/bin/env python3
"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

A/B stub reviewer for the cutover differential (tools/ab_run.sh).

Stands in for pr_review_wrapper.py on both arms: saves the payload it
was handed to $AB_PAYLOAD_DIR/{pr|issue}-N.json (sorted keys, so the
two arms' payloads are byte-comparable) and verdicts a deterministic
skip. No network, no model, no post. Ignores every argument the
callers append (--task issue, hosts, ...). Throwaway tooling: delete
with ab_run.sh/ab_diff.py after the cutover proves out.
"""

import json
import os
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    kind = "pr" if "pull_request" in payload else "issue"
    number = (payload.get("pull_request") or payload.get("issue") or {}).get("number")
    out = Path(os.environ["AB_PAYLOAD_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{number}.json").write_text(
        json.dumps(payload, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"classification": "skip", "message": ""}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
