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

Project facts are deployment data: loaded from a file and spliced
into every role prompt, so one bot serves projects with different rules."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402


class ProjectFactsTests(unittest.TestCase):
    def test_load_normalizes_trailing_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.md"
            path.write_text("##Facts:\nfact one")
            self.assertEqual(llm_prompt.load_project_facts(path), "##Facts:\nfact one\n\n")
            path.write_text("  \n\n")
            self.assertEqual(llm_prompt.load_project_facts(path), "")

    def test_facts_appear_in_every_role_prompt(self) -> None:
        facts = "##Testproject facts:\nthe build tool is frobnicate\n\n"
        for role in ("review", "combiner", "triager"):
            with self.subTest(role=role):
                prompt = llm_prompt.generate_llm_prompt(
                    role=role, vendor="openai", model="m", features=set(),
                    repo_roots=[], container_repo_mounts=[],
                    reviewer_username="fairy", project_facts=facts,
                )
                self.assertIn(facts, prompt)
                # Generic patch hygiene stays shared, not per-deployment.
                self.assertIn("Additional Minor issues:", prompt)

    def test_shipped_facts_files_load(self) -> None:
        files = sorted((REPO_ROOT / "project_facts").glob("*.md"))
        self.assertGreaterEqual(len(files), 3)
        for path in files:
            with self.subTest(path=path.name):
                facts = llm_prompt.load_project_facts(path)
                self.assertTrue(facts.startswith("##"))
                self.assertTrue(facts.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
