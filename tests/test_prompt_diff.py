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

prompt_diff --dump runs the dumped revision's own code, so it must cope
with checkouts whose ``common.setup_logging`` takes no ``color`` keyword
and which have no ``llm_prompt`` module at all: the dump succeeds and
records the missing prompt generator in an error file instead of crashing.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class PromptDiffDumpTest(unittest.TestCase):
    def test_dump_checkout_without_prompt_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkout = Path(tmpdir) / "checkout"
            checkout.mkdir()
            (checkout / "common.py").write_text(
                "def setup_logging(logger, verbose, *extra_loggers):\n"
                "    pass\n", encoding="utf-8")
            outdir = Path(tmpdir) / "prompts"
            outdir.mkdir()
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "prompt_diff.py"),
                 "--dump", str(outdir), "--checkout", str(checkout),
                 "--inputs",
                 str(REPO_ROOT / "tools" / "prompt_diff_inputs.json")],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("ERROR importing llm_prompt",
                          (outdir / "llm_prompt_error.txt")
                          .read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
