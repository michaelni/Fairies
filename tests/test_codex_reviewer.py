"""CodexReviewer: command construction, event parsing, run() behavior."""

import argparse
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codex_reviewer
import podman_host
import review_pipeline
from codex_reviewer import (
    CodexReviewer,
    CodexUsageLimit,
    build_codex_exec_command,
    summarize_codex_events,
)
from llm_review_api import BadModelOutput, ReviewContext, RoleSpec

MACHINES = (
    podman_host.parse_shell_host("fairy@h1"),
    podman_host.parse_shell_host("arm64=fairy@h2"),
)

ROLE = RoleSpec(
    name="reviewer",
    schema={"name": "review", "strict": True, "schema": {
        "type": "object",
        "properties": {"classification": {"type": "string"},
                       "message": {"type": "string"}},
        "required": ["classification", "message"],
    }},
    user_texts=lambda ctx: ["Review the following PR.", "Trailing note."],
    validate=lambda obj: dict(obj),
)


def _ctx(**kwargs) -> ReviewContext:
    defaults = dict(
        request={"pull_request": {"number": 7, "title": "t"}},
        patch_text="the patch", patch_truncated=False, source_bundle=None,
        source_files=[], source_notes=[], reviewer_username="fairy",
        ci_triage_mode=False, repo_roots=[], repo_mount_paths=["/work/ffmpeg"],
        machines=MACHINES, shell_socket_path="/run/fairy/shell.sock",
    )
    defaults.update(kwargs)
    return ReviewContext(**defaults)


class BuildCommandTests(unittest.TestCase):
    def _cmd(self, **kwargs):
        defaults = dict(
            codex_bin="codex", model="gpt-5.6-sol", effort="xhigh",
            scratch_dir="/tmp/s", schema_path="/tmp/s/schema.json",
            last_message_path="/tmp/s/last.json",
            socket_path="/run/fairy/shell.sock",
            machine_labels=("x86_64", "arm64"),
        )
        defaults.update(kwargs)
        return build_codex_exec_command(**defaults)

    def test_zero_local_execution_flag_set(self) -> None:
        cmd = self._cmd()
        joined = " ".join(cmd)
        self.assertIn("features.shell_tool=false", cmd)
        self.assertIn("--sandbox read-only", joined)
        self.assertIn("analytics.enabled=false", cmd)
        self.assertIn('web_search="cached"', cmd)
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)
        self.assertNotIn("--yolo", cmd)

    def test_prompt_comes_from_stdin(self) -> None:
        self.assertEqual("-", self._cmd()[-1])

    def test_mcp_shell_config_present_with_socket(self) -> None:
        cmd = self._cmd()
        joined = " ".join(cmd)
        self.assertIn('mcp_servers.shell.default_tools_approval_mode="approve"', cmd)
        self.assertIn("mcp_servers.shell.command=", joined)
        args_value = next(
            c for c in cmd if c.startswith("mcp_servers.shell.args="))
        bridge_args = json.loads(args_value.partition("=")[2])
        self.assertIn("--socket", bridge_args)
        self.assertEqual(
            ["x86_64", "arm64"],
            [bridge_args[i + 1] for i, a in enumerate(bridge_args)
             if a == "--machine"],
        )

    def test_no_mcp_config_without_socket(self) -> None:
        cmd = self._cmd(socket_path=None)
        self.assertFalse(any("mcp_servers" in c for c in cmd))

    def test_effort_flag_only_when_set(self) -> None:
        self.assertIn('model_reasoning_effort="xhigh"', self._cmd())
        self.assertFalse(any(
            "model_reasoning_effort" in c for c in self._cmd(effort=None)))


class EventParsingTests(unittest.TestCase):
    def test_usage_and_errors_extracted(self) -> None:
        jsonl = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "t1"}),
            "codex banner line",
            json.dumps({"type": "turn.completed",
                        "usage": {"input_tokens": 100, "output_tokens": 7}}),
            json.dumps({"type": "turn.failed",
                        "error": {"message": "boom"}}),
        ])
        usage, error_text = summarize_codex_events(jsonl)
        self.assertEqual({"input_tokens": 100, "output_tokens": 7}, usage)
        self.assertIn("boom", error_text)

    def test_tolerates_unknown_shapes(self) -> None:
        usage, error_text = summarize_codex_events("[1,2]\n{}\nnot json\n")
        self.assertEqual({}, usage)
        self.assertEqual("", error_text)


class CodexReviewerRunTests(unittest.TestCase):
    def _run(self, *, jsonl="", stderr="", returncode=0,
             last_message='{"classification": "approve", "message": "ok"}',
             ctx=None, **reviewer_kwargs):
        reviewer = CodexReviewer(
            "gpt-5.6-sol", name="codex:gpt-5.6-sol", role=ROLE,
            **reviewer_kwargs)

        def fake_run(cmd, **kwargs):
            self.last_cmd = cmd
            self.last_prompt = kwargs.get("input")
            if last_message is not None:
                path = cmd[cmd.index("--output-last-message") + 1]
                with open(path, "w", encoding="utf-8") as f:
                    f.write(last_message)
            return subprocess.CompletedProcess(
                cmd, returncode, stdout=jsonl, stderr=stderr)

        with mock.patch.object(codex_reviewer.subprocess, "run",
                               side_effect=fake_run):
            return reviewer.run(ctx or _ctx())

    def test_validated_result_from_last_message(self) -> None:
        result = self._run()
        self.assertEqual("approve", result["classification"])
        self.assertIn("the patch", self.last_prompt)
        self.assertIn("Review the following PR.", self.last_prompt)

    def test_scratch_dir_is_not_a_git_repo(self) -> None:
        # codex transmits cwd git metadata with no off switch; the scratch
        # cwd staying non-git is what keeps those fields absent.
        self._run()
        scratch = self.last_cmd[self.last_cmd.index("--cd") + 1]
        self.assertFalse(os.path.exists(os.path.join(scratch, ".git")))

    def test_usage_limit_is_hard_failure(self) -> None:
        jsonl = json.dumps({"type": "turn.failed", "error": {
            "message": "You've hit your usage limit",
            "type": "usage_limit_reached"}})
        with self.assertRaises(CodexUsageLimit):
            self._run(jsonl=jsonl, last_message=None, returncode=1)

    def test_missing_final_message_is_error_despite_rc_zero(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no final message"):
            self._run(last_message=None, returncode=0)

    def test_non_json_final_message_is_bad_model_output(self) -> None:
        with self.assertRaises(BadModelOutput):
            self._run(last_message="I approve of this patch.")

    def test_shellless_context_omits_mcp(self) -> None:
        self._run(ctx=_ctx(shell_socket_path=None))
        self.assertFalse(any("mcp_servers" in c for c in self.last_cmd))

    def test_invalid_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CodexReviewer("m", name="codex:m", role=ROLE, effort="ultra")

    def test_passes_are_serialized(self) -> None:
        # One auth.json must not serve concurrent jobs.
        import threading
        import time
        active = []

        def fake_run(cmd, **kwargs):
            active.append(1)
            self.assertEqual(1, len(active))
            time.sleep(0.02)
            active.pop()
            path = cmd[cmd.index("--output-last-message") + 1]
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"classification": "approve", "message": "ok"}')
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        reviewer = CodexReviewer("m", name="codex:m", role=ROLE)
        with mock.patch.object(codex_reviewer.subprocess, "run",
                               side_effect=fake_run):
            threads = [
                threading.Thread(target=reviewer.run, args=(_ctx(),))
                for _ in range(3)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()


class FactoryTests(unittest.TestCase):
    def test_make_reviewer_builds_codex(self) -> None:
        args = argparse.Namespace(
            codex_bin="codex-pinned", codex_timeout_seconds=0.0,
            debug_response_dir=None, podman_max_tool_rounds=0,
            podman_exec_timeout=600.0,
        )
        reviewer = review_pipeline.make_reviewer(
            "codex:gpt-5.6-sol@xhigh", args=args, resources=None,
            role=ROLE, verbose=False,
        )
        self.assertIsInstance(reviewer, CodexReviewer)
        self.assertEqual("codex:gpt-5.6-sol", reviewer.name)
        self.assertEqual("xhigh", reviewer.effort)
        self.assertEqual("codex-pinned", reviewer.codex_bin)

    def test_bad_effort_suffix_is_cli_error(self) -> None:
        args = argparse.Namespace(
            codex_bin="codex", codex_timeout_seconds=0.0,
            debug_response_dir=None, podman_max_tool_rounds=0,
            podman_exec_timeout=600.0,
        )
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "codex:gpt-5.6-sol@ultra", args=args, resources=None,
                role=ROLE, verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
