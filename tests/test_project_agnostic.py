"""Ratchet: no new project-specific bits in general code.

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
BASELINE = 14  # lower this when cleaning existing mentions up


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
