"""Project facts are deployment data: loaded from a file and spliced
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
        for role in ("reviewer", "combiner", "triager"):
            with self.subTest(role=role):
                prompt = llm_prompt.generate_llm_prompt(
                    role=role, vendor="openai", model="m", features=set(),
                    repo_roots=[], container_repo_mounts=[],
                    reviewer_username="fairy", project_facts=facts,
                )
                self.assertIn(facts, prompt)
                # Generic patch hygiene stays shared, not per-deployment.
                self.assertIn("Additional Minor issues:", prompt)

    def test_shipped_ffmpeg_facts_load(self) -> None:
        facts = llm_prompt.load_project_facts(REPO_ROOT / "project_facts" / "ffmpeg.md")
        self.assertIn("##FFmpeg project facts:", facts)
        facts = llm_prompt.load_project_facts(REPO_ROOT / "project_facts" / "ffmpeg-web.md")
        self.assertIn("ffmpeg-web", facts)


if __name__ == "__main__":
    unittest.main()
