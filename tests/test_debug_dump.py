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

dump_response_debug_artifacts: one JSONL conversation file per tool loop.

Before 2026-07-03 every tool round wrote its own ``<response id>.json``;
a single GLM review produced 30+ files (2176 files / 230MB in production
after two days), which does not scale. Rounds now append to the file the
first round created.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import dump_response_debug_artifacts  # noqa: E402


class _Resp:
    def __init__(self, rid: str) -> None:
        self.id = rid

    def model_dump(self) -> dict:
        return {"id": self.id, "output": []}


class ConversationDumpTests(unittest.TestCase):
    def test_rounds_append_to_one_conversation_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = dump_response_debug_artifacts(
                _Resp("resp_a"), {"model": "m", "input": ["round1"]},
                wrapper_request={"pull_request": {"number": 7}},
                debug_dir=tmp, verbose=False,
            )
            p2 = dump_response_debug_artifacts(
                _Resp("resp_b"), {"model": "m", "input": ["round2"]},
                wrapper_request={"pull_request": {"number": 7}},
                debug_dir=tmp, verbose=False, conversation=p1,
            )
            self.assertEqual(p1, p2)
            self.assertEqual([Path(p1)], list(Path(tmp).iterdir()))
            self.assertTrue(p1.endswith("resp_a.jsonl"))
            records = [json.loads(l) for l in Path(p1).read_text().splitlines()]
            self.assertEqual(2, len(records))
            self.assertEqual(["round1"], records[0]["request"]["input"])
            self.assertEqual("resp_b", records[1]["response"]["id"])
            for r in records:
                self.assertEqual(7, r["wrapper_request"]["pull_request"]["number"])

    def test_without_conversation_each_call_gets_its_own_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = dump_response_debug_artifacts(
                _Resp("resp_a"), {}, debug_dir=tmp, verbose=False)
            p2 = dump_response_debug_artifacts(
                _Resp("resp_b"), {}, debug_dir=tmp, verbose=False)
            self.assertNotEqual(p1, p2)
            self.assertEqual(2, len(list(Path(tmp).iterdir())))


if __name__ == "__main__":
    unittest.main()
