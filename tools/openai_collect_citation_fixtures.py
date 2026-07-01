#!/usr/bin/env python3
"""Normalize raw OpenAI debug dumps into citation fixtures.

This script scans a debug-response directory, extracts the top-level response
objects, and writes compact fixtures for citation rendering replay tests. It is
needed because the raw debug dumps contain extra wrapper metadata, while the
citation tests only need stable response payloads to exercise marker parsing
and rendering behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect OpenAI run payloads into citation functional-test fixtures.",
    )
    parser.add_argument(
        "--input-dir",
        default=".openai_debug",
        help="Directory containing raw OpenAI debug payload JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures/openai_citation_runs",
        help="Directory where normalized fixture JSON files are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of fixture files to export (0 = no limit).",
    )
    return parser.parse_args()


def normalize_fixture_name(path: Path, payload: dict[str, object]) -> str:
    response = payload.get("response")
    if isinstance(response, dict):
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in response_id)
            return f"{safe}.json"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
    return f"{safe_stem}.json"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]

    exported = 0
    skipped = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        if not isinstance(payload, dict):
            skipped += 1
            continue

        response = payload.get("response")
        if not isinstance(response, dict):
            skipped += 1
            continue

        fixture = {
            "source_file": path.name,
            "response": response,
        }
        out_name = normalize_fixture_name(path, payload)
        out_path = output_dir / out_name
        out_path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        exported += 1

    print(f"exported={exported} skipped={skipped} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
