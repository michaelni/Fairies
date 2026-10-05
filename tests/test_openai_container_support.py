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
"""
import sys
import types
import unittest
from pathlib import Path

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:  # pragma: no cover - test shim
        pass

    class _FakeError(Exception):
        pass

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.RateLimitError = _FakeError
    fake_openai.AuthenticationError = _FakeError
    fake_openai.InternalServerError = _FakeError
    sys.modules["openai"] = fake_openai

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.dotenv_values = lambda *_args, **_kwargs: {}
    sys.modules["dotenv"] = fake_dotenv

import podman_repos
import pr_review_wrapper as wrapper
import openai_reviewer


class OpenAIContainerSupportTests(unittest.TestCase):
    def test_model_visible_repo_head_map_avoids_host_paths_without_container(self) -> None:
        repo_roots = [Path("/tmp/proj/ffmpeg"), Path("/tmp/proj/all_ffmpeg")]
        repo_heads = {
            str(repo_roots[0].resolve()): "a" * 40,
            str(repo_roots[1].resolve()): "b" * 40,
        }

        visible = wrapper.build_model_visible_repo_head_map(
            repo_roots,
            repo_heads,
        )

        self.assertEqual({"ffmpeg": "a" * 40, "all_ffmpeg": "b" * 40}, visible)

    def test_model_visible_repo_head_map_local_specs_use_container_path(self) -> None:
        repo_root = Path("/tmp/ffmpeg")
        specs = [
            podman_repos.RepoSpec(
                repo_root=repo_root,
                name="ffmpeg",
                head_sha="b" * 40,
                container_path="/work/ffmpeg",
                mirror_path="fairy-mirrors/ffmpeg.git",
            )
        ]
        repo_heads = {str(repo_root.resolve()): "b" * 40}
        visible = wrapper.build_model_visible_repo_head_map(
            [repo_root],
            repo_heads,
            podman_repo_specs=specs,
        )
        self.assertEqual({"/work/ffmpeg": "b" * 40}, visible)

    def test_build_response_tools_shell_reuses_given_container(self) -> None:
        tools = openai_reviewer.build_response_tools(
            vector_store_ids=["vs_1"],
            file_search_max_num_results=7,
            use_web_search=False,
            web_search_context_size="medium",
            web_search_cache_only=False,
            web_search_domains=[],
            use_shell=True,
            shell_container_id="cntr_123",
        )

        self.assertEqual("file_search", tools[0]["type"])
        self.assertEqual("shell", tools[1]["type"])
        self.assertEqual("container_reference", tools[1]["environment"]["type"])
        self.assertEqual("cntr_123", tools[1]["environment"]["container_id"])
        self.assertEqual("code_interpreter", tools[2]["type"])

    def test_build_response_tools_default_code_interpreter_is_auto(self) -> None:
        tools = openai_reviewer.build_response_tools(
            vector_store_ids=[],
            file_search_max_num_results=None,
            use_web_search=False,
            web_search_context_size="medium",
            web_search_cache_only=False,
            web_search_domains=[],
            use_shell=False,
            shell_container_id=None,
        )

        self.assertEqual([{"type": "code_interpreter", "container": {"type": "auto"}}], tools)

    def test_build_response_tools_podman_shell_is_function_only_no_code_interpreter(self) -> None:
        tools = openai_reviewer.build_response_tools(
            vector_store_ids=[],
            file_search_max_num_results=None,
            use_web_search=False,
            web_search_context_size="medium",
            web_search_cache_only=False,
            web_search_domains=[],
            use_shell=False,
            shell_container_id=None,
            use_podman_shell=True,
        )
        self.assertEqual(1, len(tools))
        self.assertEqual("function", tools[0]["type"])
        self.assertEqual("shell", tools[0]["name"])

    def test_prompt_cache_key_for(self) -> None:
        # Built from forge data only: upload file_ids change per retry and
        # must never enter the key.
        self.assertEqual(
            "fairy:review:22290:32486a55dc96",
            openai_reviewer.prompt_cache_key_for(
                "review",
                {"pull_request": {"number": 22290,
                                  "head_sha": "32486a55dc96e1e177ddc58f2f4971785f138554"}}),
        )
        self.assertEqual(
            "fairy:issue_investigator:7",
            openai_reviewer.prompt_cache_key_for("issue_investigator", {"issue": {"number": 7}}),
        )
        self.assertIsNone(openai_reviewer.prompt_cache_key_for("review", {}))

    def test_build_response_include_omits_code_interpreter_for_podman_shell(self) -> None:
        inc = openai_reviewer.build_response_include(
            vector_store_ids=[],
            use_web_search=False,
            use_podman_shell=True,
        )
        self.assertNotIn("code_interpreter_call.outputs", inc)


if __name__ == "__main__":
    unittest.main()
