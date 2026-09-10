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

The podman shell cookbook adapts to the mounted repos: the PR-refs
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
        role="review", vendor="openai", model="m",
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

    def test_devices_spec_toggles_devices_line(self) -> None:
        self.assertNotIn("host devices", _prompt(["ffmpeg"]))
        self.assertIn(
            "the host devices /dev/snd/controlC10, /dev/snd/pcmC10D1c passed through",
            _prompt(["ffmpeg"], [_machine(devices=("/dev/snd/controlC10", "/dev/snd/pcmC10D1c"))]),
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
        self.assertIn("Choose arm64 if you want to test arm/arm64 or NEON!", text)
        self.assertIn("Choose x86_64 if you want to test on x86/x86-64,", text)
        self.assertNotIn(
            "Choose", _prompt(["ffmpeg"], [_machine("big"), _machine("small")]))

    def test_foreign_deployment_makes_no_ffmpeg_claims(self) -> None:
        text = _prompt(["somerepo"])
        self.assertIn("In the somerepo checkout every pull request's head", text)
        self.assertNotIn("aggregates project data as subtrees", text)
        self.assertNotIn("FATE sample-suite snapshot", text)

    def test_recoll_advertised_only_with_checkouts(self) -> None:
        text = _prompt(["ffmpeg", "all_ffmpeg"])
        self.assertIn("recollq", text)
        # the index is a snapshot the model cannot refresh
        self.assertIn("does not reflect anything you check out", text)
        self.assertIn("not regexes or substrings", text)
        bare = llm_prompt.generate_llm_prompt(
            role="review", vendor="openai", model="m",
            features={"podman_shell"}, repo_roots=[],
            container_repo_mounts=[], reviewer_username="fairy",
            machines=[_machine()],
        )
        self.assertNotIn("recollq", bare)


if __name__ == "__main__":
    unittest.main()
