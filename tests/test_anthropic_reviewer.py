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

AnthropicReviewer Messages tool-loop, replayed with a scripted client.

The ``anthropic`` SDK is mocked (like the OpenAI replay tests mock
``openai``) so the test runs without the package or network. It pins the
shape the loop depends on: the model investigates via the ``shell`` tool
(dispatched onto a fake ContainerShellSession), then returns its verdict by
calling ``submit_review`` with REVIEW_SCHEMA-shaped input.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Inject a minimal fake ``anthropic`` before importing the reviewer.
if "anthropic" not in sys.modules:
    fake = types.ModuleType("anthropic")

    class _E(Exception):
        pass

    class APIConnectionError(_E):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class RateLimitError(_E):
        pass

    class InternalServerError(_E):
        pass

    class OverloadedError(_E):
        pass

    fake.Anthropic = object  # replaced per-test via _client
    fake.APIConnectionError = APIConnectionError
    fake.APITimeoutError = APITimeoutError
    fake.RateLimitError = RateLimitError
    fake.InternalServerError = InternalServerError
    fake.OverloadedError = OverloadedError
    sys.modules["anthropic"] = fake

import podman_host  # noqa: E402
from llm_review_api import ReviewContext  # noqa: E402
import anthropic_reviewer  # noqa: E402
import llm_prompt  # noqa: E402


class _Block:
    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


class _Message:
    def __init__(self, content: list[_Block]) -> None:
        self.content = content


class _ScriptedClient:
    """Returns the next queued _Message on each messages.create call."""

    def __init__(self, scripted: list[_Message]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs: object) -> _Message:
        self.calls.append(dict(kwargs))
        return self._scripted.pop(0)


class _FakeShell:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    def exec(self, command: str, *, cwd: str | None = None, timeout_s: float = 120.0):
        self.commands.append(command)
        return podman_host.ExecResult(
            exit_code=0, stdout="commit deadbeef\n", stderr="",
            duration_s=0.01, stdout_truncated=False, stderr_truncated=False,
        )

    def close(self) -> None:
        self.closed = True


def _ctx(shell: _FakeShell | None) -> ReviewContext:
    return ReviewContext(
        request={"pull_request": {"number": 7, "title": "t"}, "reviewer_username": "fairy"},
        patch_text="===== BEGIN PATCH =====\n...\n",
        patch_truncated=False,
        source_bundle="file a.c\n...",
        source_files=["a.c"],
        source_notes=[],
        reviewer_username="fairy",
        ci_triage_mode=False,
        repo_roots=[Path.cwd()],
        repo_mount_paths=["/work/ffmpeg"],
        machines=[podman_host.ShellHostSpec(
            "x86_64", podman_host.RemoteHost("fairy@h"))] if shell is not None else [],
        open_shell=(lambda label: (shell, "")) if shell is not None else None,
    )


class AnthropicReviewLoopTests(unittest.TestCase):
    def test_shell_round_then_submit(self) -> None:
        shell = _FakeShell()
        client = _ScriptedClient([
            _Message([_Block(type="tool_use", id="t1", name="shell",
                             input={"command": "git log -1"})]),
            _Message([_Block(type="tool_use", id="t2", name="submit_review",
                             input={"classification": "minor_issues_approve",
                                    "message": "LLM review: one nit.",
                                    "head_vs_branch_diff_evidence": False})]),
        ])
        reviewer = anthropic_reviewer.AnthropicReviewer(
            "glm-4.6", name="zai:glm-4.6",
            base_url="https://api.z.ai/api/anthropic", api_key_env="ZAI_API_KEY",
        )
        reviewer._client = lambda: client  # type: ignore[method-assign]

        review = reviewer.review(_ctx(shell))

        self.assertEqual("minor_issues_approve", review.classification)
        self.assertEqual("LLM review: one nit.", review.message)
        self.assertEqual("zai:glm-4.6", review.model)
        self.assertEqual(["git log -1"], shell.commands)
        self.assertFalse(shell.closed)
        self.assertEqual(2, len(client.calls))
        # The second request must replay the assistant tool_use turn plus the
        # tool_result, so the model can act on the shell output.
        second_msgs = client.calls[1]["messages"]
        self.assertEqual("assistant", second_msgs[1]["role"])
        self.assertEqual("tool_use", second_msgs[1]["content"][0]["type"])
        self.assertEqual("tool_result", second_msgs[2]["content"][0]["type"])

    def test_cache_breakpoints_move_to_newest_block(self) -> None:
        # Regression: without cache_control the Anthropic API caches
        # nothing and re-bills the full context every round (probed
        # 2026-07-14 on claude-haiku: t2 in=7745 read=0 bare vs in=3
        # read=7730 marked; z.ai accepts the markers and is unaffected).
        shell = _FakeShell()
        client = _ScriptedClient([
            _Message([_Block(type="tool_use", id="t1", name="shell",
                             input={"command": "git log -1"})]),
            _Message([_Block(type="tool_use", id="t2", name="submit_review",
                             input={"classification": "approve", "message": "",
                                    "head_vs_branch_diff_evidence": False})]),
        ])
        reviewer = anthropic_reviewer.AnthropicReviewer("claude-opus-4", name="anthropic:claude-opus-4")
        reviewer._client = lambda: client  # type: ignore[method-assign]
        reviewer.review(_ctx(shell))

        marker = {"type": "ephemeral"}
        for call in client.calls:
            self.assertEqual(marker, call["system"][0]["cache_control"])
        # The recorded calls share block dicts, so only the final marker
        # position is observable: exactly one, on the newest block (the
        # round-2 tool_result), earlier markers stripped.
        final = client.calls[1]["messages"]
        marked = [b for m in final if isinstance(m["content"], list)
                  for b in m["content"] if "cache_control" in b]
        self.assertEqual(1, len(marked))
        self.assertEqual(marker, marked[0]["cache_control"])
        self.assertIs(final[-1]["content"][-1], marked[0])
        self.assertEqual("tool_result", marked[0]["type"])

    def test_mark_cache_breakpoint_moves_marker(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]},
            {"role": "user", "content": "plain string nudge"},
        ]
        anthropic_reviewer._mark_cache_breakpoint(messages)
        self.assertEqual({"type": "ephemeral"},
                         messages[0]["content"][-1]["cache_control"])
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]})
        anthropic_reviewer._mark_cache_breakpoint(messages)
        self.assertNotIn("cache_control", messages[0]["content"][-1])
        self.assertEqual({"type": "ephemeral"},
                         messages[2]["content"][0]["cache_control"])

    def test_module_logger_names_match_wrapper_registration(self) -> None:
        # pr_review_wrapper.main() attaches log handlers to these
        # loggers BY NAME because the modules are imported lazily. A module
        # rename would silently detach their logs again (regression: GLM
        # reviewer activity was invisible in run logs, 2026-07-02).
        import anthropic_common
        import shell_tool
        self.assertEqual("anthropic_reviewer", anthropic_reviewer.logger.name)
        self.assertEqual("anthropic_common", anthropic_common.logger.name)
        self.assertEqual("shell_tool", shell_tool.logger.name)

    def test_no_shell_direct_submit(self) -> None:
        client = _ScriptedClient([
            _Message([_Block(type="tool_use", id="t1", name="submit_review",
                             input={"classification": "approve", "message": "",
                                    "head_vs_branch_diff_evidence": False})]),
        ])
        reviewer = anthropic_reviewer.AnthropicReviewer("claude-opus-4", name="anthropic:claude-opus-4")
        reviewer._client = lambda: client  # type: ignore[method-assign]

        review = reviewer.review(_ctx(None))

        self.assertEqual("approve", review.classification)
        # submit_review only (no shell tool) when the context has no shell.
        self.assertEqual(1, len(client.calls[0]["tools"]))

    def test_triager_role_via_submit_review(self) -> None:
        from llm_prompt import make_triager_role

        client = _ScriptedClient([
            _Message([_Block(type="tool_use", id="t1", name="submit_review",
                             input={"route": "engage", "message": "", "reason": "new code",
                                    "prompt_injection": False,
                                    "requested_verbosity": None})]),
        ])
        role = make_triager_role(allowed_models=[], allowed_labels=[])
        reviewer = anthropic_reviewer.AnthropicReviewer(
            "glm-5.3", name="zai:glm-5.3", role=role,
        )
        reviewer._client = lambda: client  # type: ignore[method-assign]

        result = reviewer.run(_ctx(None))

        self.assertEqual("engage", result["route"])
        self.assertEqual("submit_review", client.calls[0]["tools"][0]["name"])
        self.assertIn("route", client.calls[0]["tools"][0]["input_schema"]["properties"])


class EffortThinkingTests(unittest.TestCase):
    def _submit(self) -> _Message:
        return _Message([_Block(type="tool_use", id="t1", name="submit_review",
                                input={"classification": "approve", "message": "",
                                       "head_vs_branch_diff_evidence": False})])

    def _run(self, effort: str | None, reasoning_summary: str | None = None) -> dict:
        client = _ScriptedClient([self._submit()])
        reviewer = anthropic_reviewer.AnthropicReviewer(
            "glm-5.3", name="zai:glm-5.3", effort=effort,
            reasoning_summary=reasoning_summary,
        )
        reviewer._client = lambda: client  # type: ignore[method-assign]
        reviewer.review(_ctx(None))
        return client.calls[0]

    def test_default_sends_no_thinking_parameter(self) -> None:
        self.assertNotIn("thinking", self._run(None))

    def test_off_disables_thinking(self) -> None:
        self.assertEqual({"type": "disabled"}, self._run("off")["thinking"])

    def test_reasoning_summary_requests_summarized_display(self) -> None:
        # claude-opus-5 returns a thinking summary only with
        # display=summarized (probed 2026-08-09); z.ai glm-5.3 accepts it
        # and still returns thinking (probed 2026-08-15), so the same
        # request shape serves both backends.
        for summary in ("concise", "detailed"):
            call = self._run("high", summary)
            self.assertEqual(
                {"type": "adaptive", "display": "summarized"}, call["thinking"])
            self.assertEqual({"effort": "high"}, call["output_config"])

    def test_reasoning_summary_auto_keeps_provider_default(self) -> None:
        self.assertEqual({"type": "adaptive"}, self._run("high", "auto")["thinking"])

    def test_reasoning_summary_without_effort_sends_no_thinking(self) -> None:
        self.assertNotIn("thinking", self._run(None, "detailed"))

    def test_effort_sets_adaptive_thinking(self) -> None:
        # The named-effort dialect (thinking adaptive + output_config.effort)
        # is what current Claude models require (manual budget_tokens gets a
        # 400 there) and what z.ai's Anthropic endpoint honors for GLM
        # (probed 2026-07-03: thinking volume scales low < high < max).
        for effort in ("low", "xhigh", "max"):
            call = self._run(effort)
            self.assertEqual({"type": "adaptive"}, call["thinking"])
            self.assertEqual({"effort": effort}, call["output_config"])
            self.assertEqual(
                anthropic_reviewer.DEFAULT_ANTHROPIC_MAX_TOKENS, call["max_tokens"],
            )

    def test_unknown_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            anthropic_reviewer.AnthropicReviewer("glm-5.3", name="zai:glm-5.3", effort="turbo")

    def test_client_gets_explicit_timeout(self) -> None:
        # Regression (production 2026-07-03, PR 22592): with the SDK-default
        # client timeout, anthropic's static pre-flight rejects non-streaming
        # requests whose max_tokens could take >10 min to generate
        # ("Streaming is required for operations that may take longer than
        # 10 minutes", anthropic 0.115.0 _calculate_nonstreaming_timeout,
        # threshold max_tokens > 128000*600/3600 = 21333); every GLM draft
        # died before a single request when max_tokens crossed it. An
        # explicit client timeout opts out of that pre-flight.
        recorded: dict = {}

        class _Recorder:
            def __init__(self, **kwargs: object) -> None:
                recorded.update(kwargs)

        with mock.patch.object(anthropic_reviewer, "Anthropic", _Recorder), \
                mock.patch.object(anthropic_reviewer, "load_api_key",
                                  return_value="k"):
            anthropic_reviewer.AnthropicReviewer(
                "glm-5.3", name="zai:glm-5.3", effort="medium",
            )._client()
        self.assertEqual(
            anthropic_reviewer.ANTHROPIC_TIMEOUT_S, recorded["timeout"],
        )

    def test_thinking_blocks_are_replayed_with_signature(self) -> None:
        # The Messages API rejects tool-result turns unless the assistant's
        # thinking blocks are echoed unmodified (signature included).
        shell = _FakeShell()
        client = _ScriptedClient([
            _Message([
                _Block(type="thinking", thinking="check the log", signature="sig1"),
                _Block(type="tool_use", id="t1", name="shell",
                       input={"command": "git log -1"}),
            ]),
            self._submit(),
        ])
        reviewer = anthropic_reviewer.AnthropicReviewer(
            "glm-5.3", name="zai:glm-5.3", effort="medium",
        )
        reviewer._client = lambda: client  # type: ignore[method-assign]
        reviewer.review(_ctx(shell))
        echoed = client.calls[1]["messages"][1]["content"]
        self.assertEqual(
            {"type": "thinking", "thinking": "check the log", "signature": "sig1"},
            echoed[0],
        )
        self.assertEqual("tool_use", echoed[1]["type"])


class PersistBranchesTests(unittest.TestCase):
    """When collection is wired up, the reviewer hands its own shell
    sessions and its validated branch declarations and pull_requests to
    ctx.collect_branches, and the collected records land on the Review."""

    PR_REQUEST = {"repo": "ffmpeg", "branch": "fix-x", "title": "Fix x",
                  "body": "", "target": "master"}
    DECLARATION = {"repo": "ffmpeg", "branch": "fix-y", "action": "push"}

    def _scripted(self) -> _ScriptedClient:
        return _ScriptedClient([
            _Message([_Block(type="tool_use", id="t1", name="shell",
                             input={"command": "git push fairy fix-x"})]),
            _Message([_Block(type="tool_use", id="t2", name="submit_review",
                             input={"classification": "minor_issues_approve",
                                    "message": "LLM review: built fix-x.",
                                    "head_vs_branch_diff_evidence": False,
                                    "branches": [self.DECLARATION],
                                    "pull_requests": [self.PR_REQUEST]})]),
        ])

    def _reviewer(self) -> anthropic_reviewer.AnthropicReviewer:
        reviewer = anthropic_reviewer.AnthropicReviewer(
            "claude-opus-4", name="anthropic:claude-opus-4",
            role=llm_prompt.role_with_branches(llm_prompt.REVIEWER_ROLE,
                                               ["ffmpeg"]))
        reviewer._client = lambda: self._scripted()  # type: ignore[method-assign]
        return reviewer

    def test_collect_receives_sessions_and_result_lands_on_review(self) -> None:
        shell = _FakeShell()
        collected = dict(branch="fix-x", mode="ff", pr=self.PR_REQUEST,
                         repo="ffmpeg", sha="a" * 40, old_sha=None,
                         bundle="", objects_repo="/client/ffmpeg",
                         diff_base_sha=None)
        calls: list[tuple] = []

        def collect(sessions, declared, pull_requests):
            calls.append((list(sessions), declared, pull_requests))
            return [collected]

        ctx = _ctx(shell)
        ctx.collect_branches = collect
        review = self._reviewer().review(ctx)
        self.assertEqual((collected,), review.branches)
        self.assertEqual(calls,
                         [([shell], [self.DECLARATION], [self.PR_REQUEST])])

    def test_without_collection_no_branches_are_minted(self) -> None:
        review = self._reviewer().review(_ctx(_FakeShell()))
        self.assertEqual((), review.branches)


if __name__ == "__main__":
    unittest.main()
