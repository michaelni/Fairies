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

Branch persistence's declaration vocabulary: the ``branches`` and
``pull_requests`` verdict lists, their boundary sanitizers, and the
prompt/role plumbing.
"""

from __future__ import annotations

import base64
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Minimal fakes so ``openai_reviewer`` imports without the SDK (same
# pattern as test_wrapper_functional_replay).
try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    _fake_openai = types.ModuleType("openai")

    class _FakeOpenAIError(Exception):
        pass

    _fake_openai.OpenAI = object
    _fake_openai.RateLimitError = _FakeOpenAIError
    _fake_openai.AuthenticationError = _FakeOpenAIError
    _fake_openai.InternalServerError = _FakeOpenAIError
    sys.modules["openai"] = _fake_openai
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    _fake_dotenv = types.ModuleType("dotenv")
    _fake_dotenv.dotenv_values = lambda *_args, **_kwargs: {}
    sys.modules["dotenv"] = _fake_dotenv

try:
    import openai_reviewer  # noqa: E402
except ModuleNotFoundError:  # a transitive dependency (e.g. httpx) is absent
    openai_reviewer = None

import llm_prompt  # noqa: E402
import llm_review_api  # noqa: E402
import podman_host  # noqa: E402
from llm_review_api import (  # noqa: E402
    MAX_PERSIST_BRANCHES,
    sanitize_branch_declarations,
    sanitize_pull_requests,
    schema_with_branches,
    validate_result_with_branches,
)

REPOS = ["ffmpeg", "ffmpeg-web"]
VALID_PR_REQUEST = {"repo": "ffmpeg", "branch": "pr7-fix-overflow",
                    "title": "Fix overflow", "body": "details",
                    "target": "master"}
VALID_DECLARATION = {"repo": "ffmpeg", "branch": "pr7-fix-overflow",
                     "action": "push"}


class SanitizePullRequestsTests(unittest.TestCase):
    def test_valid_requests_survive(self) -> None:
        self.assertEqual(sanitize_pull_requests([VALID_PR_REQUEST], REPOS),
                         [VALID_PR_REQUEST])

    def test_unknown_repo_is_dropped(self) -> None:
        self.assertEqual(
            sanitize_pull_requests(
                [dict(VALID_PR_REQUEST, repo="all_ffmpeg")], REPOS), [])

    def test_unsafe_branch_names_are_dropped(self) -> None:
        for name in ("a b", "../evil", "refs/heads/x", "-fast", "", "a" * 65,
                     "ünïcode", "x;rm", None, 7):
            with self.subTest(name=name):
                self.assertEqual(
                    sanitize_pull_requests(
                        [dict(VALID_PR_REQUEST, branch=name)], REPOS), [])

    def test_bad_title_or_target_is_dropped(self) -> None:
        for bad in (dict(VALID_PR_REQUEST, title=""),
                    dict(VALID_PR_REQUEST, target="-evil"),
                    dict(VALID_PR_REQUEST, target="a b"),
                    "open one please"):
            with self.subTest(bad=bad):
                self.assertEqual(sanitize_pull_requests([bad], REPOS), [])

    def test_duplicates_and_cap(self) -> None:
        many = [dict(VALID_PR_REQUEST, branch=f"b{i}")
                for i in range(MAX_PERSIST_BRANCHES + 2)]
        self.assertEqual(len(sanitize_pull_requests(many, REPOS)),
                         MAX_PERSIST_BRANCHES)
        self.assertEqual(
            len(sanitize_pull_requests([VALID_PR_REQUEST] * 2, REPOS)), 1)

    def test_non_list_is_empty(self) -> None:
        for raw in (None, "x", {"branch": "b"}):
            self.assertEqual(sanitize_pull_requests(raw, REPOS), [])


class SanitizeBranchDeclarationsTests(unittest.TestCase):
    def test_valid_declarations_survive(self) -> None:
        self.assertEqual(
            sanitize_branch_declarations([VALID_DECLARATION], REPOS),
            [VALID_DECLARATION])

    def test_bad_action_repo_or_name_is_dropped(self) -> None:
        for bad in (dict(VALID_DECLARATION, action="merge"),
                    dict(VALID_DECLARATION, repo="all_ffmpeg"),
                    dict(VALID_DECLARATION, branch="a b"),
                    {"branch": "x"}, "keep it"):
            with self.subTest(bad=bad):
                self.assertEqual(sanitize_branch_declarations([bad], REPOS), [])

    def test_duplicates_and_cap(self) -> None:
        many = [dict(VALID_DECLARATION, branch=f"b{i}")
                for i in range(MAX_PERSIST_BRANCHES + 2)]
        self.assertEqual(len(sanitize_branch_declarations(many, REPOS)),
                         MAX_PERSIST_BRANCHES)
        self.assertEqual(
            len(sanitize_branch_declarations([VALID_DECLARATION] * 2, REPOS)), 1)


class PersistenceSchemaTests(unittest.TestCase):
    def test_schema_accepts_valid_output(self) -> None:
        schema = schema_with_branches(llm_review_api.REVIEW_SCHEMA, REPOS)
        llm_review_api.check_schema(
            {"classification": "approve", "message": "",
             "head_vs_branch_diff_evidence": False,
             "branches": [VALID_DECLARATION],
             "pull_requests": [VALID_PR_REQUEST]},
            schema["schema"])

    def test_schema_rejects_unknown_repo_and_keys(self) -> None:
        schema = schema_with_branches(llm_review_api.REVIEW_SCHEMA, REPOS)
        for bad in (dict(VALID_PR_REQUEST, repo="all_ffmpeg"),
                    dict(VALID_PR_REQUEST, remote="origin")):
            with self.subTest(bad=bad):
                with self.assertRaises(llm_review_api.SchemaError):
                    llm_review_api.check_schema(
                        {"classification": "approve", "message": "",
                         "head_vs_branch_diff_evidence": False,
                         "branches": [],
                         "pull_requests": [bad]},
                        schema["schema"])

    def test_base_schema_still_rejects_pull_requests(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {"classification": "approve", "message": "",
                 "head_vs_branch_diff_evidence": False, "pull_requests": []},
                llm_review_api.REVIEW_SCHEMA["schema"])


class ValidateWithBranchesTests(unittest.TestCase):
    def test_composes_over_labels_validator(self) -> None:
        result = validate_result_with_branches(
            {"classification": "minor_issues_approve", "message": "m",
             "head_vs_branch_diff_evidence": False,
             "label_changes": [{"label": "important", "op": "add",
                                "reason": "r", "post": False}],
             "branches": [dict(VALID_DECLARATION, branch="cleanup",
                               action="delete"),
                          dict(VALID_DECLARATION, action="merge")],
             "pull_requests": [VALID_PR_REQUEST,
                               dict(VALID_PR_REQUEST, branch="bad name")]},
            REPOS,
            lambda obj: llm_review_api.validate_review_result(obj, ["important"]))
        self.assertEqual(result["classification"], "minor_issues_approve")
        self.assertEqual(len(result["label_changes"]), 1)
        self.assertEqual(result["branches"],
                         [dict(VALID_DECLARATION, branch="cleanup",
                               action="delete")])
        self.assertEqual(result["pull_requests"], [VALID_PR_REQUEST])

    def test_verdict_fields_stay_strict(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            validate_result_with_branches(
                {"message": "m", "pull_requests": []}, REPOS,
                llm_review_api.validate_review)


GOOD_RECORD = {
    "branch": "pr7-fix-overflow", "mode": "ff", "pr": None, "repo": "ffmpeg",
    "sha": "a" * 40, "old_sha": None,
    "bundle": base64.b64encode(b"BUNDLE").decode(),
    "objects_repo": "/client/ffmpeg", "diff_base_sha": "b" * 40,
}
FORCE_RECORD = dict(GOOD_RECORD, mode="force", old_sha="d" * 40)
DELETE_RECORD = {
    "branch": "pr7-stale", "mode": "delete", "pr": None, "repo": "ffmpeg",
    "sha": "c" * 40, "old_sha": "c" * 40, "bundle": "",
    "objects_repo": "/client/ffmpeg", "diff_base_sha": None,
}


class RoleWithBranchesTests(unittest.TestCase):
    def test_composes_over_role_with_labels(self) -> None:
        role = llm_prompt.role_with_branches(
            llm_prompt.role_with_labels(llm_prompt.REVIEWER_ROLE, ["important"]),
            REPOS)
        props = role.schema["schema"]["properties"]
        self.assertIn("branches", props)
        self.assertIn("pull_requests", props)
        self.assertIn("label_changes", props)
        self.assertTrue(role.prompt_kwargs["persist_branches"])
        result = role.validate(
            {"classification": "approve", "message": "",
             "head_vs_branch_diff_evidence": False,
             "label_changes": [], "branches": [VALID_DECLARATION],
             "pull_requests": [VALID_PR_REQUEST]})
        self.assertEqual(result["branches"], [VALID_DECLARATION])
        self.assertEqual(result["pull_requests"], [VALID_PR_REQUEST])

    def test_prompt_section_appears_only_when_enabled(self) -> None:
        kwargs = dict(
            vendor="anthropic", model="m", features=set(),
            repo_roots=[], container_repo_mounts=["/work/ffmpeg"],
            reviewer_username="fairy")
        for role in ("review", "combiner", "issue_investigator", "issue_combiner"):
            with self.subTest(role=role):
                self.assertNotIn("##Persisting branches",
                                 llm_prompt.generate_llm_prompt(role=role, **kwargs))
                enabled = llm_prompt.generate_llm_prompt(
                    role=role, persist_branches=True, **kwargs)
                self.assertIn("##Persisting branches", enabled)
                self.assertIn('remote "fairy"', enabled)
                self.assertIn("fairy/<name>", enabled)

    def test_combiner_prompt_says_only_declared_branches_survive(self) -> None:
        prompt = llm_prompt.prompt_persist_branches(
            llm_prompt.PromptFor("combiner", "m"))
        self.assertIn("on your fairy remotes", prompt)
        self.assertIn("only your own declarations count", prompt)
        self.assertIn("Re-declare", prompt)


@unittest.skipUnless(openai_reviewer is not None,
                     "openai_reviewer dependencies missing")
class OpenAIPersistBranchesTests(unittest.TestCase):
    """The OpenAI reviewer hands its shared per-machine sessions and its
    validated declarations and pull_requests to ctx.collect_branches,
    whose records land on the Review."""

    def test_collects_via_shared_sessions(self) -> None:
        session = object()
        resources = openai_reviewer.OpenAIResources(
            client=None, tools=[], include=[], patch_file_id=None,
            vector_store_ids=[], shared_container_id=None,
            shells={"x86_64": session},
            open_shell=lambda label: (session, ""),
            uploaded_file_ids=[], debug_dir_specified=False)
        args = SimpleNamespace(
            podman=True, verbose=False, verbosity=None, service_tier=None,
            top_p=None, reasoning_summary=None, max_tool_calls=None,
            max_output_tokens=1000, web_search="off",
            podman_max_tool_rounds=0, podman_exec_timeout=60.0,
            podman_parallel_tool_calls=False, debug_response_dir=".dbg",
            use_openai_container_repos=False, model="gpt-5.4")
        reviewer = openai_reviewer.OpenAIReviewer(
            args, resources, model="gpt-5.4",
            role=llm_prompt.role_with_branches(llm_prompt.REVIEWER_ROLE, REPOS),
            service_tier=None)
        calls: list[tuple] = []

        def collect(sessions, declared, pull_requests):
            calls.append((list(sessions), declared, pull_requests))
            return [GOOD_RECORD]

        ctx = llm_review_api.ReviewContext(
            request={"pull_request": {"number": 7}},
            patch_text="", patch_truncated=False, source_bundle=None,
            source_files=[], source_notes=[], reviewer_username="fairy",
            ci_triage_mode=False, repo_roots=[], repo_mount_paths=["/work/ffmpeg"],
            machines=[podman_host.ShellHostSpec(
                "x86_64", podman_host.RemoteHost("fairy@h"))],
            collect_branches=collect)
        payload = json.dumps({
            "classification": "minor_issues_approve", "message": "LLM review",
            "head_vs_branch_diff_evidence": False,
            "branches": [VALID_DECLARATION],
            "pull_requests": [VALID_PR_REQUEST]})
        with mock.patch.object(openai_reviewer,
                               "run_responses_resolving_podman_shell",
                               return_value={}), \
                mock.patch.object(openai_reviewer, "extract_response_annotations",
                                  return_value=[]), \
                mock.patch.object(openai_reviewer,
                                  "extract_response_file_citation_metadata",
                                  return_value={}), \
                mock.patch.object(openai_reviewer, "extract_response_text",
                                  return_value=payload):
            review = reviewer.review(ctx)
        self.assertEqual(review.branches, (GOOD_RECORD,))
        self.assertEqual(calls, [([session], [VALID_DECLARATION],
                                  [VALID_PR_REQUEST])])


