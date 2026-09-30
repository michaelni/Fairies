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

A pull request whose head is fairy's own published branch is reported
to the model as that branch, so the model updates it instead of
opening a replacement. Fixture: the head of FFmpeg #24798, opened by
fairy from ``Forgejo_Fairy:fairy/issue23991-bugfix``.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import llm_prompt  # noqa: E402

PR = {
    "number": 24798,
    "title": "avfilter/buffersrc: reject invalid video pixel formats",
    "user": {"login": "Forgejo_Fairy"},
    "head": {
        "ref": "fairy/issue23991-bugfix",
        "sha": "0b4c1f2a6d8e9c3b5a7f1e2d4c6b8a0f9e7d5c3b",
        "repo": {"full_name": "Forgejo_Fairy/FFmpeg"},
    },
    "base": {"ref": "master"},
}

SPEC = fairy.parse_branch_push_spec(
    "ffmpeg=FFmpeg/FFmpeg=https://forge.example.org/Forgejo_Fairy/FFmpeg.git")


def _args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{
        "owner": "FFmpeg", "repo": "FFmpeg", "branch_push": [SPEC],
        "branch_push_head_owner": ["ffmpeg=Forgejo_Fairy"],
        "llm_review_cmd": "./wrapper", "triage_labels": [],
        "llm_max_patch_bytes": 10_000, "verbose": 0, "llm_timeout": None,
        **overrides})


class OwnFairyBranchTests(unittest.TestCase):
    def test_head_in_fork_is_the_container_branch(self) -> None:
        self.assertEqual(fairy.own_fairy_branch(_args(), PR),
                         {"repo": "ffmpeg", "branch": "issue23991-bugfix"})

    def test_other_fork_with_fairy_prefix_is_not_hers(self) -> None:
        pr = {**PR, "head": {**PR["head"], "repo": {"full_name": "alice/FFmpeg"}}}
        self.assertIsNone(fairy.own_fairy_branch(_args(), pr))

    def test_branch_outside_fairy_namespace_is_not_hers(self) -> None:
        pr = {**PR, "head": {**PR["head"], "ref": "master"}}
        self.assertIsNone(fairy.own_fairy_branch(_args(), pr))

    def test_fork_owner_defaults_to_the_forge_owner(self) -> None:
        self.assertIsNone(fairy.own_fairy_branch(
            _args(branch_push_head_owner=[]), PR))

    def test_deleted_fork_is_not_hers(self) -> None:
        pr = {**PR, "head": {**PR["head"], "repo": None}}
        self.assertIsNone(fairy.own_fairy_branch(_args(), pr))


class PayloadTests(unittest.TestCase):
    def _payload(self, args: SimpleNamespace) -> dict:
        captured: dict[str, str] = {}

        def fake_run_cmd(_cmd, *, input_text: str, **_kw):
            captured["input_text"] = input_text
            return SimpleNamespace(
                returncode=0,
                stdout='{"classification": "approve", "message": ""}')

        with (
            patch.object(fairy, "patch_shas_for_run",
                         return_value=("base", "head")),
            patch.object(fairy, "fetch_patch_for_llm",
                         return_value=("patch", False)),
            patch.object(fairy, "run_cmd", side_effect=fake_run_cmd),
        ):
            fairy.run_llm_review(args, PR, "no", [], "Forgejo_Fairy")
        return json.loads(captured["input_text"])

    def test_own_pull_request_carries_head_repo_and_branch(self) -> None:
        pr = self._payload(_args())["pull_request"]
        self.assertEqual(pr["head_repo"], "Forgejo_Fairy/FFmpeg")
        self.assertEqual(pr["fairy_branch"],
                         {"repo": "ffmpeg", "branch": "issue23991-bugfix"})

    def test_foreign_pull_request_has_no_fairy_branch(self) -> None:
        pr = self._payload(_args(branch_push=[]))["pull_request"]
        self.assertEqual(pr["head_repo"], "Forgejo_Fairy/FFmpeg")
        self.assertNotIn("fairy_branch", pr)


class UserTextTests(unittest.TestCase):
    def _metadata(self, pull_request: dict) -> dict:
        text = llm_prompt.make_user_text(
            {"pull_request": pull_request, "discussion": []}, [], [], False)
        start = text.index("Pull request metadata:\n") + len("Pull request metadata:\n")
        return json.loads(text[start:text.index("\n\nPull request body:")])

    def test_fairy_branch_reaches_the_metadata(self) -> None:
        info = self._metadata({"number": 24798,
                               "head_repo": "Forgejo_Fairy/FFmpeg",
                               "fairy_branch": {"repo": "ffmpeg",
                                                "branch": "issue23991-bugfix"}})
        self.assertEqual(info["head_repo"], "Forgejo_Fairy/FFmpeg")
        self.assertEqual(info["fairy_branch"]["branch"], "issue23991-bugfix")

    def test_metadata_without_fairy_branch_omits_it(self) -> None:
        self.assertNotIn("fairy_branch", self._metadata({"number": 1}))


if __name__ == "__main__":
    unittest.main()
