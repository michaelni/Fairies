"""The podman shell cookbook adapts to the mounted repos: the PR-refs
line names the primary checkout, and the all_ffmpeg / FATE lines appear
only when those repos are actually part of the deployment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_prompt  # noqa: E402


def _prompt(repo_names: list[str]) -> str:
    return llm_prompt.generate_llm_prompt(
        role="reviewer", vendor="openai", model="m",
        features={"podman_shell"},
        repo_roots=[Path("/x") / name for name in repo_names],
        container_repo_mounts=[f"/work/{name}" for name in repo_names],
        reviewer_username="fairy",
    )


class RepoCookbookTests(unittest.TestCase):
    def test_ffmpeg_deployment_keeps_full_cookbook(self) -> None:
        text = _prompt(["ffmpeg", "all_ffmpeg"])
        self.assertIn("In the ffmpeg checkout every pull request's head", text)
        self.assertIn("FATE sample-suite snapshot", text)
        self.assertIn("aggregates project data as subtrees", text)

    def test_web_deployment_names_its_checkout_and_drops_fate(self) -> None:
        text = _prompt(["ffmpeg-web", "all_ffmpeg"])
        self.assertIn("In the ffmpeg-web checkout every pull request's head", text)
        self.assertNotIn("FATE sample-suite snapshot", text)
        self.assertIn("aggregates project data as subtrees", text)

    def test_foreign_deployment_makes_no_ffmpeg_claims(self) -> None:
        text = _prompt(["somerepo"])
        self.assertIn("In the somerepo checkout every pull request's head", text)
        self.assertNotIn("aggregates project data as subtrees", text)
        self.assertNotIn("FATE sample-suite snapshot", text)


if __name__ == "__main__":
    unittest.main()
