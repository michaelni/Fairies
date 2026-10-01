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

AgyReviewer: command construction, stream-json parsing, run() behavior."""

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agy_reviewer
import cli_reviewer
import podman_host
import review_pipeline
from agy_reviewer import (
    TOKEN_NAME,
    AgyReviewer,
    AgyTurnFailed,
    build_agy_command,
    exhausted_bucket,
    summarize_agy_events,
)
from cli_reviewer import CliAuthFailed, CliUsageLimit
from llm_review_api import BadModelOutput, ProviderTurnFailed
from test_cli_container import ROLE, FakeCliContainer, make_ctx

AGY_HOST = podman_host.parse_shell_host("fairy@agybox")

# from an agy 1.2.12 --output-format stream-json run of a
# 493 KB review prompt with --json-schema, 2026-09-28 (model
# gemini-3.8-flash-low); the echoed json_schema is cut, the
# conversation_id a dummy
_REAL_RESULT_EVENT = (
    '{"event":"result","result":{"conversation_id":"00000000-0000-0000-0000-000000000001",'
    '"status":"SUCCESS","response":"{\\"classification\\":\\"major_issues\\",'
    '\\"head_vs_branch_diff_evidence\\":false,\\"label_changes\\":[],\\"message\\":\\"The patch reads one past the buffer end.\\"}\\n",'
    '"duration_seconds":16.463901763,"num_turns":1,'
    '"structured_output":{"classification":"major_issues","head_vs_branch_diff_evidence":false,'
    '"label_changes":[],"message":"The patch reads one past the buffer end."},'
    '"json_schema":{"type":"object"},'
    '"usage":{"input_tokens":105711,"output_tokens":3621,"thinking_tokens":3504,'
    '"cache_read_tokens":0,"total_tokens":109332}}}\n'
)
_REAL_STREAM = (
    '{"event":"init","cwd":"/tmp","tools":[],"permission_mode":"default"}\n'
    '{"event":"step_update","conversation_id":"00000000","step_index":0,"state":"DONE","step_type":"PLANNER_RESPONSE"}\n'
    + _REAL_RESULT_EVENT
)

# agy 1.2.12 --output-format json envelopes, 2026-09-28, with a dummy
# conversation_id, wrapped as the stream-json result event the docs give
# the same shape (https://antigravity.google/docs/cli/headless/)
_DENIED_ENVELOPE = (
    '{"conversation_id":"00000000-0000-0000-0000-000000000002","status":"SUCCESS",'
    '"response":"","duration_seconds":3.800255879,"num_turns":1,'
    '"usage":{"input_tokens":26465,"output_tokens":115,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":26580},'
    '"denied_actions":[{"action":"mcp","display_name":"CallMcpTool"}]}'
)
_AUTH_FAILED_ENVELOPE = (
    '{"conversation_id":"","status":"ERROR","response":"","error":"authentication failed or timed out",'
    '"duration_seconds":0,"num_turns":0,'
    '"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":0}}'
)
_AUTH_FAILED_STDERR = (
    "Authentication required. Please visit the URL to log in:\n"
    "  https://accounts.google.com/o/oauth2/auth?...\n\n"
    "Waiting for authentication (timeout 60s)...\n"
    "Or, paste the authorization code here and press Enter:\n"
    "Error: authentication timed out.\n"
    "error: authentication failed or timed out\n"
)


# verbatim agy 1.2.12 `agy -p /usage --output-format json`, 2026-09-28,
# descriptions cut
_USAGE_ENVELOPE = (
    '{"conversation_id":"","status":"SUCCESS","response":"Gemini Models\\tWeekly Limit Remaining\\t100%\\t2026-10-05T02:49:45Z\\n",'
    '"duration_seconds":0,"num_turns":0,'
    '"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":0},'
    '"command":{"name":"usage","data":{"groups":['
    '{"name":"Gemini Models","buckets":['
    '{"id":"gemini-weekly","name":"Weekly Limit Remaining","window":"weekly","remaining_fraction":0.999922513961792,"reset_time":"2026-10-05T02:49:45Z"},'
    '{"id":"gemini-5h","name":"Five Hour Limit Remaining","window":"5h","remaining_fraction":0.999535322189331,"reset_time":"2026-09-28T07:49:45Z"}]},'
    '{"name":"Claude and GPT models","buckets":['
    '{"id":"3p-weekly","name":"Weekly Limit Remaining","window":"weekly","remaining_fraction":1,"reset_time":"2026-10-05T02:49:52Z"},'
    '{"id":"3p-5h","name":"Five Hour Limit Remaining","window":"5h","remaining_fraction":1,"reset_time":"2026-09-28T07:49:52Z"}]}]}}}\n'
)


def _usage_with(bucket_id: str, remaining: float) -> str:
    envelope = json.loads(_USAGE_ENVELOPE)
    for group in envelope["command"]["data"]["groups"]:
        for bucket in group["buckets"]:
            if bucket["id"] == bucket_id:
                bucket["remaining_fraction"] = remaining
    return json.dumps(envelope) + "\n"


def _result_event(envelope: str) -> str:
    return json.dumps({"event": "result", "result": json.loads(envelope)}) + "\n"


def _agy_home(token='{"refresh_token": "r", "expiry": "2027-01-01T00:00:00Z"}') -> str:
    home = tempfile.mkdtemp(prefix="agy-home-")
    if token is not None:
        Path(home, TOKEN_NAME).write_text(token, encoding="utf-8")
    return home


class BuildCommandTests(unittest.TestCase):
    def _cmd(self, **kwargs):
        defaults = dict(agy_bin="agy", model="gemini-3.1-pro-high",
                        effort="high", schema_path="/work/.agy-run/schema.json")
        defaults.update(kwargs)
        return build_agy_command(**defaults)

    def test_headless_stream_json_with_schema(self) -> None:
        cmd = self._cmd()
        joined = " ".join(cmd)
        self.assertEqual("agy", cmd[0])
        self.assertIn("--input-format stream-json", joined)
        self.assertIn("--output-format stream-json", joined)
        self.assertIn("--json-schema /work/.agy-run/schema.json", joined)
        self.assertIn("--model gemini-3.1-pro-high", joined)
        self.assertIn("--disable-slash-commands", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertNotIn("-p", cmd)

    def test_effort_flag_only_when_set(self) -> None:
        self.assertIn("--effort high", " ".join(self._cmd()))
        self.assertNotIn("--effort", self._cmd(effort=None))


class EventParsingTests(unittest.TestCase):
    def test_usage_and_result_from_the_last_result_event(self) -> None:
        usage, result = summarize_agy_events(_REAL_STREAM)
        self.assertEqual(105711, usage["input_tokens"])
        self.assertEqual("SUCCESS", result["status"])
        self.assertEqual("major_issues", result["structured_output"]["classification"])

    def test_no_result_event(self) -> None:
        self.assertEqual(({}, None), summarize_agy_events(
            '{"event":"init"}\nnot json\n[1]\n{"event":"result","result":"x"}\n'))

    def test_exhausted_bucket_is_the_models_group_at_zero(self) -> None:
        self.assertIsNone(exhausted_bucket(_USAGE_ENVELOPE, "gemini-3.1-pro-high"))
        drained = _usage_with("gemini-5h", 0)
        self.assertEqual("gemini-5h", exhausted_bucket(drained, "gemini-3.1-pro-high")["id"])
        self.assertIsNone(exhausted_bucket(drained, "claude-opus-4-6-thinking"))
        self.assertEqual("3p-weekly", exhausted_bucket(
            _usage_with("3p-weekly", 0), "gpt-oss-120b-medium")["id"])
        self.assertIsNone(exhausted_bucket(_usage_with("3p-weekly", 0), "other-model"))
        self.assertIsNone(exhausted_bucket("not json\n{}\n", "gemini-3.1-pro-high"))


class AgyReviewerRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._relay_sessions = []

    def _reviewer(self, *, agy_home=None, **kw):
        return AgyReviewer("gemini-3.1-pro-high", name="agy:gemini-3.1-pro-high",
                           role=ROLE, host=AGY_HOST,
                           home=agy_home or _agy_home(), **kw)

    def _run(self, *, stdout=_REAL_STREAM, stderr="", returncode=0,
             ctx=None, reviewer=None, refreshed_token=None,
             usage=_USAGE_ENVELOPE):
        reviewer = reviewer or self._reviewer()
        proc = subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)
        self.container = FakeCliContainer(
            proc, None, credential_name=TOKEN_NAME, refreshed_auth=refreshed_token,
            later_run_results=[subprocess.CompletedProcess([], 0, stdout=usage, stderr="")])
        with mock.patch.object(cli_reviewer, "CliContainer",
                               return_value=self.container), \
                mock.patch.object(cli_reviewer, "CliShellRelay") as relay:
            relay.return_value.start.return_value = relay.return_value
            relay.return_value.opened_sessions.return_value = self._relay_sessions
            return reviewer.run(ctx or make_ctx())

    def _poisoned_after(self, exc_type, **run_kw) -> list:
        self._relay_sessions = [mock.Mock(spec=podman_host.ContainerShellSession)]
        poisoned = []
        with self.assertRaises(exc_type):
            self._run(ctx=make_ctx(report_poisoned=poisoned.append), **run_kw)
        return poisoned

    def test_structured_output_is_the_validated_result(self) -> None:
        result = self._run()
        self.assertEqual("major_issues", result["classification"])

    def test_prompt_goes_in_as_one_stream_json_user_message(self) -> None:
        self._run()
        (line,) = self.container.input_text.splitlines()
        message = json.loads(line)
        self.assertEqual("user", message["event"])
        self.assertIn("the patch", message["message"]["content"])
        self.assertIn("Review the following PR.", message["message"]["content"])

    def test_container_gets_token_settings_mcp_config_and_bridge(self) -> None:
        self._run()
        copied = self.container.copied
        for name in (TOKEN_NAME, "output_schema.json", "mcp_config.json",
                     "settings.json", "mcp_shell_bridge.py",
                     "shell_bridge_client.py", "relay.py", "hooks.json",
                     agy_reviewer.HOOK_SCRIPT):
            self.assertIn(name, copied)
        (server,) = json.loads(copied["mcp_config.json"])["mcpServers"].values()
        self.assertEqual("python3", server["command"])
        self.assertEqual("/work/.agy-run/mcp_shell_bridge.py", server["args"][0])
        self.assertIn("--socket", server["args"])
        self.assertEqual(["x86_64", "arm64"],
                         [server["args"][i + 1] for i, a in enumerate(server["args"])
                          if a == "--machine"])
        self.assertEqual({"permissions": {"allow": ["mcp(shell/*)"]}},
                         json.loads(copied["settings.json"]))

    def test_container_env_is_the_private_home_without_self_update(self) -> None:
        self._run()
        self.assertEqual({"HOME": agy_reviewer.CONTAINER_HOME,
                          "AGY_CLI_DISABLE_AUTO_UPDATE": "true"}, self.container.env)

    def test_shellless_context_grants_nothing(self) -> None:
        self._run(ctx=make_ctx(open_shell=None))
        self.assertNotIn("mcp_config.json", self.container.copied)
        self.assertNotIn("settings.json", self.container.copied)
        self.assertIn("hooks.json", self.container.copied)
        self.assertIn("answer from the text given",
                      self.container.copied[agy_reviewer.HOOK_SCRIPT])
        self.assertNotIn("MCP tool shell", self.container.input_text)

    def test_native_tools_hook_denies_with_the_mcp_shell_as_reason(self) -> None:
        self._run()
        (group,) = json.loads(self.container.copied["hooks.json"])[
            "fairy-native-tools"]["PreToolUse"]
        self.assertRegex("run_command", group["matcher"])
        self.assertRegex("write_to_file", group["matcher"])
        self.assertRegex("browser_click_element", group["matcher"])
        self.assertNotRegex("view_file", group["matcher"])
        self.assertNotRegex("call_mcp_tool", group["matcher"])
        (hook,) = group["hooks"]
        self.assertEqual(f"sh ~/.gemini/config/{agy_reviewer.HOOK_SCRIPT}",
                         hook["command"])
        script = self.container.copied[agy_reviewer.HOOK_SCRIPT]
        self.assertTrue(script.startswith("#!/bin/sh\n"))
        printed = subprocess.run(["sh", "-c", script], input="{}", capture_output=True,
                                 text=True, check=True).stdout
        decision = json.loads(printed)
        self.assertEqual("deny", decision["decision"])
        self.assertIn("MCP tool shell", decision["reason"])
        self.assertIn("MCP tool shell", self.container.input_text)

    def test_denied_tool_without_output_ends_the_turn(self) -> None:
        with self.assertRaises(AgyTurnFailed) as caught:
            self._run(stdout=_result_event(_DENIED_ENVELOPE))
        self.assertIsInstance(caught.exception, ProviderTurnFailed)
        self.assertIn("CallMcpTool", str(caught.exception))

    def test_denied_tool_with_output_is_kept_and_logged(self) -> None:
        result = json.loads(_REAL_RESULT_EVENT)
        result["result"]["denied_actions"] = [{"action": "command"}]
        with self.assertLogs(agy_reviewer.logger, "WARNING"):
            self.assertEqual("major_issues",
                             self._run(stdout=json.dumps(result))["classification"])

    def test_error_status_ends_the_turn_when_quota_is_left(self) -> None:
        envelope = json.loads(_AUTH_FAILED_ENVELOPE)
        envelope["error"] = "internal error"
        with self.assertRaisesRegex(AgyTurnFailed, "internal error"):
            self._run(stdout=_result_event(json.dumps(envelope)), returncode=1)
        (main_cmd, usage_cmd) = self.container.run_cmds
        self.assertEqual(["agy", "-p", "/usage", "--output-format", "json"], usage_cmd)

    def test_error_status_with_the_models_quota_at_zero_is_a_usage_limit(self) -> None:
        envelope = json.loads(_AUTH_FAILED_ENVELOPE)
        envelope["error"] = "internal error"
        with self.assertRaisesRegex(CliUsageLimit, "gemini-3.1-pro-high"):
            self._run(stdout=_result_event(json.dumps(envelope)), returncode=1,
                      usage=_usage_with("gemini-weekly", 0))
        self.assertEqual([], self._poisoned_after(
            CliUsageLimit, stdout="", returncode=1,
            usage=_usage_with("gemini-weekly", 0)))

    def test_a_successful_run_never_queries_usage(self) -> None:
        self._run()
        self.assertEqual(1, len(self.container.run_cmds))

    def test_auth_prompt_sets_the_token_aside(self) -> None:
        """agy prints the same login prompt for a dead token and for an
        unreachable Google (both observed on 1.2.12); every such run is
        treated as a dead token."""
        home = _agy_home()
        for stdout in (_result_event(_AUTH_FAILED_ENVELOPE), ""):
            Path(home, TOKEN_NAME).write_text("{}", encoding="utf-8")
            with self.subTest(stdout=stdout), self.assertRaises(CliAuthFailed), \
                    self.assertLogs(cli_reviewer.logger, "CRITICAL"):
                self._run(reviewer=self._reviewer(agy_home=home), stdout=stdout,
                          stderr=_AUTH_FAILED_STDERR, returncode=1)
            self.assertFalse(Path(home, TOKEN_NAME).exists())
            self.assertTrue(Path(home, TOKEN_NAME + ".invalid").is_file())

    def test_refreshed_token_is_persisted_from_agy_home(self) -> None:
        home = _agy_home()
        rotated = '{"refresh_token": "rotated", "expiry": "2028-01-01T00:00:00Z"}'
        self._run(reviewer=self._reviewer(agy_home=home), refreshed_token=rotated)
        self.assertEqual(rotated, Path(home, TOKEN_NAME).read_text(encoding="utf-8"))
        self.assertTrue(Path(home, TOKEN_NAME + ".bak").is_file())
        (read_back,) = [p for p, _ in self.container.read_file_calls]
        self.assertEqual(f"{agy_reviewer.CONTAINER_HOME}/.gemini/antigravity-cli/"
                         f"{TOKEN_NAME}", read_back)

    def test_no_result_event_is_a_crash_that_poisons(self) -> None:
        poisoned = self._poisoned_after(RuntimeError, stdout="", returncode=137)
        self.assertEqual(1, len(poisoned))

    def test_denied_turn_and_bad_output_do_not_poison(self) -> None:
        self.assertEqual([], self._poisoned_after(
            AgyTurnFailed, stdout=_result_event(_DENIED_ENVELOPE)))
        result = json.loads(_REAL_RESULT_EVENT)
        del result["result"]["structured_output"]
        self.assertEqual([], self._poisoned_after(
            BadModelOutput, stdout=json.dumps(result)))

    def test_debug_dump_is_prefixed_agy(self) -> None:
        debug_dir = tempfile.mkdtemp(prefix="agy-dbg-")
        self._run(reviewer=self._reviewer(debug_dir=debug_dir))
        (path,) = Path(debug_dir).iterdir()
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertRegex(payload["response"]["id"], r"^agy-\d{8}-\d{6}-review$")
        self.assertEqual(105711, payload["response"]["usage"]["input_tokens"])
        self.assertEqual(self.container.cmd, payload["request"]["cmd"])

    def test_missing_token_is_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, f"{TOKEN_NAME} not found"):
            self._run(reviewer=self._reviewer(agy_home=_agy_home(token=None)))
        self.assertIsNone(self.container.cmd)

    def test_no_agy_host_is_error(self) -> None:
        reviewer = AgyReviewer("m", name="agy:m", role=ROLE, home=_agy_home())
        with self.assertRaisesRegex(RuntimeError, "requires --agy-host"):
            reviewer.run(make_ctx())

    def test_invalid_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgyReviewer("m", name="agy:m", role=ROLE, effort="xhigh")


class FactoryTests(unittest.TestCase):
    def _args(self, **extra):
        base = dict(
            agy_bin="agy-pinned", agy_timeout_seconds=0.0, agy_home=None,
            agy_host=AGY_HOST, agy_image="img:test",
            podman_exec_timeout=600.0, debug_response_dir=None,
        )
        base.update(extra)
        return argparse.Namespace(**base)

    def test_make_reviewer_builds_agy(self) -> None:
        reviewer = review_pipeline.make_reviewer(
            "agy:claude-opus-4-6-thinking@max", args=self._args(),
            resources=None, role=ROLE, verbose=False)
        self.assertIsInstance(reviewer, AgyReviewer)
        self.assertEqual("agy:claude-opus-4-6-thinking", reviewer.name)
        self.assertEqual("max", reviewer.effort)
        self.assertEqual("agy-pinned", reviewer.cli_bin)
        self.assertEqual(AGY_HOST, reviewer.host)
        self.assertEqual("img:test", reviewer.image)

    def test_agy_home_option_selects_the_login(self) -> None:
        reviewer = review_pipeline.make_reviewer(
            "agy:gemini-3.1-pro-high:agy-home=/second-home", args=self._args(),
            resources=None, role=ROLE, verbose=False)
        self.assertEqual(Path("/second-home", TOKEN_NAME), reviewer.credential_local)
        self.assertEqual("agy:gemini-3.1-pro-high+second-home", reviewer.name)

    def test_codex_home_option_is_not_an_agy_option(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "agy:gemini-3.1-pro-high:codex-home=/x", args=self._args(),
                resources=None, role=ROLE, verbose=False)

    def test_agy_without_host_is_cli_error(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "agy:gemini-3.1-pro-high", args=self._args(agy_host=None),
                resources=None, role=ROLE, verbose=False)

    def test_bad_effort_suffix_is_cli_error(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "agy:gemini-3.1-pro-high@xhigh", args=self._args(),
                resources=None, role=ROLE, verbose=False)


if __name__ == "__main__":
    unittest.main()
