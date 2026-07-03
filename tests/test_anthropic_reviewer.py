"""AnthropicReviewer Messages tool-loop, replayed with a scripted client.

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
        new_shell=(lambda: shell) if shell is not None else None,
    )


class AnthropicReviewLoopTests(unittest.TestCase):
    def test_shell_round_then_submit(self) -> None:
        shell = _FakeShell()
        client = _ScriptedClient([
            _Message([_Block(type="tool_use", id="t1", name="shell",
                             input={"command": "git log -1"})]),
            _Message([_Block(type="tool_use", id="t2", name="submit_review",
                             input={"classification": "minor_issues_approve",
                                    "message": "LLM review: one nit."})]),
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
        self.assertTrue(shell.closed)
        self.assertEqual(2, len(client.calls))
        # The second request must replay the assistant tool_use turn plus the
        # tool_result, so the model can act on the shell output.
        second_msgs = client.calls[1]["messages"]
        self.assertEqual("assistant", second_msgs[1]["role"])
        self.assertEqual("tool_use", second_msgs[1]["content"][0]["type"])
        self.assertEqual("tool_result", second_msgs[2]["content"][0]["type"])

    def test_module_logger_names_match_wrapper_registration(self) -> None:
        # openai_pr_review_wrapper.main() attaches log handlers to these
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
                             input={"classification": "ok_approve", "message": ""})]),
        ])
        reviewer = anthropic_reviewer.AnthropicReviewer("claude-opus-4", name="anthropic:claude-opus-4")
        reviewer._client = lambda: client  # type: ignore[method-assign]

        review = reviewer.review(_ctx(None))

        self.assertEqual("ok_approve", review.classification)
        # submit_review only (no shell tool) when the context has no shell.
        self.assertEqual(1, len(client.calls[0]["tools"]))


class EffortThinkingTests(unittest.TestCase):
    def _submit(self) -> _Message:
        return _Message([_Block(type="tool_use", id="t1", name="submit_review",
                                input={"classification": "ok_approve", "message": ""})])

    def _run(self, effort: str | None) -> dict:
        client = _ScriptedClient([self._submit()])
        reviewer = anthropic_reviewer.AnthropicReviewer(
            "glm-5.2", name="zai:glm-5.2", effort=effort,
        )
        reviewer._client = lambda: client  # type: ignore[method-assign]
        reviewer.review(_ctx(None))
        return client.calls[0]

    def test_default_sends_no_thinking_parameter(self) -> None:
        self.assertNotIn("thinking", self._run(None))

    def test_off_disables_thinking(self) -> None:
        self.assertEqual({"type": "disabled"}, self._run("off")["thinking"])

    def test_effort_sets_budget_and_grows_max_tokens(self) -> None:
        call = self._run("low")
        budget = anthropic_reviewer.EFFORT_THINKING_BUDGETS["low"]
        self.assertEqual({"type": "enabled", "budget_tokens": budget}, call["thinking"])
        self.assertEqual(
            anthropic_reviewer.DEFAULT_ANTHROPIC_MAX_TOKENS + budget, call["max_tokens"],
        )

    def test_unknown_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            anthropic_reviewer.AnthropicReviewer("glm-5.2", name="zai:glm-5.2", effort="xhigh")

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
            "glm-5.2", name="zai:glm-5.2", effort="medium",
        )
        reviewer._client = lambda: client  # type: ignore[method-assign]
        reviewer.review(_ctx(shell))
        echoed = client.calls[1]["messages"][1]["content"]
        self.assertEqual(
            {"type": "thinking", "thinking": "check the log", "signature": "sig1"},
            echoed[0],
        )
        self.assertEqual("tool_use", echoed[1]["type"])


if __name__ == "__main__":
    unittest.main()
