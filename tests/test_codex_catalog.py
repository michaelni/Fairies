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

codex_catalog: catalog hardening."""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codex_catalog


class HardenCatalogTests(unittest.TestCase):
    CATALOG = {
        "etag": "abc",
        "models": [
            {"slug": "gpt-5.6-sol", "input_modalities": ["text", "image"],
             "apply_patch_tool_type": "freeform", "tool_mode": "code_mode_only",
             "context_window": 400000},
            {"slug": "gpt-5.5", "input_modalities": ["text", "image"],
             "apply_patch_tool_type": "freeform", "tool_mode": None},
        ],
    }

    def test_strips_direct_host_file_tools(self) -> None:
        out = codex_catalog.harden_codex_catalog(self.CATALOG)
        for m in out["models"]:
            self.assertEqual(["text"], m["input_modalities"])  # view_image inert
            self.assertIsNone(m["apply_patch_tool_type"])      # no apply_patch
        # unrelated fields survive
        self.assertEqual(400000, out["models"][0]["context_window"])
        self.assertEqual("abc", out["etag"])

    def test_tool_mode_left_untouched(self) -> None:
        out = codex_catalog.harden_codex_catalog(self.CATALOG)
        self.assertEqual("code_mode_only", out["models"][0]["tool_mode"])
        self.assertIsNone(out["models"][1]["tool_mode"])

    def test_does_not_mutate_input(self) -> None:
        codex_catalog.harden_codex_catalog(self.CATALOG)
        first = self.CATALOG["models"][0]
        self.assertEqual(["text", "image"], first["input_modalities"])
        self.assertEqual("freeform", first["apply_patch_tool_type"])


if __name__ == "__main__":
    unittest.main()
