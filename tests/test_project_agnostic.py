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

Ratchet: no new project-specific bits in general code.

rules/project-agnostic.mdc bans project bits outside the prompt,
configuration, and per-deployment scripts. The 2026-07-14 survey found
14 lines mentioning ffmpeg in general *.py code, kept for now; this
test only allows that count to shrink.
"""
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = 14  # increasing this number is strictly forbidden; lower it when cleaning existing mentions up


@unittest.skipUnless(shutil.which("git"), "git required")
class ProjectAgnosticRatchetTests(unittest.TestCase):
    def test_no_new_ffmpeg_mentions_in_general_code(self) -> None:
        files = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        hits = [
            f"{name}:{i}: {line.strip()}"
            for name in files
            if not name.startswith("tests/") and name != "llm_prompt.py"
            for i, line in enumerate(
                (REPO_ROOT / name).read_text().splitlines(), 1)
            if "ffmpeg" in line.lower()
        ]
        self.assertLessEqual(
            len(hits), BASELINE,
            "new project-specific mention(s) in general code; see "
            "rules/project-agnostic.mdc:\n" + "\n".join(hits),
        )


if __name__ == "__main__":
    unittest.main()
