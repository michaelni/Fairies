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

A Responses API 400 whose error code names a content flag is raised as
``ProviderContentFlagged`` so the stage switches to the configured
fallback reviewer; any other 400 propagates unchanged.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import openai  # noqa: E402

import llm_prompt  # noqa: E402
import llm_review_api  # noqa: E402
import openai_reviewer  # noqa: E402
import podman_host  # noqa: E402


def bad_request(code: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {{'code': '{code}'}}}}",
        response=httpx.Response(400, request=request),
        body={"message": "refused", "type": "invalid_request", "code": code})


class ContentFlagTests(unittest.TestCase):
    def run_reviewer_with(self, exc: Exception) -> None:
        resources = openai_reviewer.OpenAIResources(
            client=None, tools=[], include=[], patch_file_id=None,
            vector_store_ids=[], uploaded_file_ids=[], debug_dir_specified=False)
        args = SimpleNamespace(
            podman=True, verbose=False, verbosity=None, service_tier=None,
            top_p=None, reasoning_summary=None, max_tool_calls=None,
            max_output_tokens=1000, web_search="off",
            podman_max_tool_rounds=0, podman_exec_timeout=60.0,
            podman_parallel_tool_calls=False, debug_response_dir=".dbg",
            model="gpt-6-sol")
        reviewer = openai_reviewer.OpenAIReviewer(
            args, resources, model="gpt-6-sol", role=llm_prompt.REVIEWER_ROLE,
            service_tier=None)
        ctx = llm_review_api.ReviewContext(
            request={"pull_request": {"number": 7}},
            patch_text="", patch_truncated=False, source_bundle=None,
            source_files=[], source_notes=[], reviewer_username="fairy",
            ci_triage_mode=False, repo_roots=[], repo_mount_paths=["/work/ffmpeg"],
            machines=[podman_host.ShellHostSpec(
                "x86_64", podman_host.RemoteHost("fairy@h"))],
            open_shell=lambda label: (object(), ""),
            close_shells=lambda sessions: None)
        with mock.patch.object(openai_reviewer, "run_responses_resolving_podman_shell",
                               side_effect=exc):
            reviewer.run(ctx)

    def test_content_flag_codes_switch_to_the_fallback(self) -> None:
        for code in openai_reviewer.CONTENT_FLAG_ERROR_CODES:
            with self.subTest(code=code):
                with self.assertRaises(llm_review_api.ProviderContentFlagged) as caught:
                    self.run_reviewer_with(bad_request(code))
                self.assertIn(code, str(caught.exception))

    def test_other_bad_requests_propagate(self) -> None:
        with self.assertRaises(openai.BadRequestError):
            self.run_reviewer_with(bad_request("unsupported_parameter"))


if __name__ == "__main__":
    unittest.main()
