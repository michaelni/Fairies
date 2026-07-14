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
import podman_host  # noqa: E402


def _machine(label: str = "x86_64", **kw: object) -> podman_host.ShellHostSpec:
    return podman_host.ShellHostSpec(label, podman_host.RemoteHost("fairy@h"), **kw)


def _prompt(
    repo_names: list[str],
    machines: list[podman_host.ShellHostSpec] | None = None,
) -> str:
    return llm_prompt.generate_llm_prompt(
        role="reviewer", vendor="openai", model="m",
        features={"podman_shell"},
        repo_roots=[Path("/x") / name for name in repo_names],
        container_repo_mounts=[f"/work/{name}" for name in repo_names],
        reviewer_username="fairy",
        machines=machines if machines is not None else [_machine()],
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

    def test_gpu_spec_toggles_gpu_line(self) -> None:
        self.assertNotIn("NVIDIA GPU", _prompt(["ffmpeg"]))
        self.assertIn(
            "NVIDIA GPU",
            _prompt(["ffmpeg"], [_machine(gpu="nvidia.com/gpu=0")]),
        )

    def test_single_machine_keeps_flat_sentence(self) -> None:
        text = _prompt(["ffmpeg"])
        self.assertIn(
            "You have 8 x86_64 CPU cores, 8g memory and tens of GB of "
            "SSD-backed disk space at your disposal.", text)
        self.assertNotIn("``machine`` parameter", text)

    def test_two_machines_render_bullets(self) -> None:
        text = _prompt(["ffmpeg"], [
            _machine(),
            _machine("arm64", cpus="12", memory="16g"),
        ])
        self.assertIn("named by its ``machine`` parameter (default x86_64)", text)
        self.assertIn("- x86_64: 8 x86_64 CPU cores, 8g memory", text)
        self.assertIn("- arm64: 12 arm64 CPU cores, 16g memory", text)
        self.assertIn("state does not carry over", text)

    def test_foreign_deployment_makes_no_ffmpeg_claims(self) -> None:
        text = _prompt(["somerepo"])
        self.assertIn("In the somerepo checkout every pull request's head", text)
        self.assertNotIn("aggregates project data as subtrees", text)
        self.assertNotIn("FATE sample-suite snapshot", text)


if __name__ == "__main__":
    unittest.main()
