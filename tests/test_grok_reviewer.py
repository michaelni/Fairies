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

GrokReviewer: command construction, result parsing, run() behavior."""

import argparse
import json
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cli_reviewer
import grok_reviewer
import podman_host
import review_pipeline
from cli_reviewer import CliAuthFailed
from grok_reviewer import (
    CREDENTIAL_NAME,
    GrokReviewer,
    GrokTurnFailed,
    build_grok_command,
    result_event,
)
from llm_review_api import ProviderTurnFailed
from test_cli_container import ROLE, FakeCliContainer, make_ctx

GROK_HOST = podman_host.parse_shell_host("fairy@grokbox")

# the init and result lines of a grok 1.0.44 run of GrokReviewer's command
# line, 2026-10-01 (model grok-4.7), as the debug dump recorded them but
# with dummy IDs
_REAL_STREAM = (
    r'{"apiKeySource":"oauth","cwd":"/work","mcp_servers":[{"name":"shell","status":"pending"}],"model":"g'
    r'rok-4.7","permissionMode":"bypassPermissions","session_id":"00000000-0000-0000-0000-000000000001","s'
    r'kills":[],"slash_commands":["compact","always-approve","context","session-info","feedback","goal"],"'
    r'subtype":"init","tools":["search_tool","use_tool"],"type":"system","uuid":"00000000-0000-0000-0000-0'
    r'00000000009"}' "\n"
    r'{"duration_api_ms":21339,"duration_ms":21403,"is_error":false,"modelUsage":{"grok-4.7-build":{"cache'
    r'CreationInputTokens":0,"cacheReadInputTokens":30848,"costUSD":0.01347148,"inputTokens":8166,"outputT'
    r'okens":1311,"webSearchRequests":0}},"num_turns":5,"result":"{\"classification\":\"reply_no_verdict\"'
    r',\"message\":\"GROK-4.7\\n\\n<!-- Scope: no commit diff or PR metadata was available in the workspac'
    r'e to review; the review machine identity was checked with uname -a. -->\\n\\nreview-machine-7f3a Lin'
    r'ux 6.8.0\"}","session_id":"00000000-0000-0000-0000-000000000001","stop_reason":"end_turn","structure'
    r'd_output":{"classification":"reply_no_verdict","message":"GROK-4.7\n\n<!-- Scope: no commit diff or '
    r'PR metadata was available in the workspace to review; the review machine identity was checked with u'
    r'name -a. -->\n\nreview-machine-7f3a Linux 6.8.0"},"subtype":"success","total_cost_usd":0.01347148,"t'
    r'ype":"result","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":30848,"input_tokens'
    r'":8166,"output_tokens":1311,"server_tool_use":{"web_search_requests":0}},"uuid":"00000000-0000-0000-'
    r'0000-000000000010"}' "\n"
)

# grok 1.0.44 result lines with dummy IDs, 2026-10-01: a run under
# --permission-mode dontAsk whose model called a tool no rule allowed,
# a run without auth.json, a run whose auth.json held made-up tokens,
# and a run with --model grok-9-nonexistent
_CANCELLED_RESULT = (
    r'{"type":"result","subtype":"error_during_execution","is_error":true,"duration_ms":31961,'
    r'"duration_api_ms":31901,"num_turns":5,"stop_reason":"cancelled","total_cost_usd":0.02064888,'
    r'"usage":{"input_tokens":15719,"output_tokens":1949,"cache_read_input_tokens":35200,'
    r'"cache_creation_input_tokens":0,"server_tool_use":{"web_search_requests":0}},'
    r'"modelUsage":{"grok-4.7-build":{"inputTokens":15719,"outputTokens":1949,"cacheReadInputTokens":35200,'
    r'"cacheCreationInputTokens":0,"webSearchRequests":0,"costUSD":0.02064888}},"errors":["cancelled"],'
    r'"session_id":"00000000-0000-0000-0000-000000000002","uuid":"00000000-0000-0000-0000-000000000003"}' "\n"
)
_NOT_SIGNED_IN_RESULT = (
    r'{"type":"result","subtype":"error_during_execution","is_error":true,"duration_ms":0,"duration_api_ms":0,'
    r'"num_turns":0,"stop_reason":null,"total_cost_usd":0.0,"usage":{"input_tokens":0,"output_tokens":0,'
    r'"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"server_tool_use":{"web_search_requests":0}},'
    r'"modelUsage":{},"errors":["Not signed in. To authenticate without a browser, run:\n  grok login '
    r'--device-code\n\nAlternatively, set the XAI_API_KEY environment variable or run `grok login` on a '
    r'machine with a browser."],"session_id":"","uuid":"00000000-0000-0000-0000-000000000004"}' "\n"
)
_REJECTED_TOKENS_RESULT = (
    r'{"type":"result","subtype":"error_during_execution","is_error":true,"duration_ms":358,"duration_api_ms":0,'
    r'"num_turns":0,"stop_reason":null,"total_cost_usd":0.0,"usage":{"input_tokens":0,"output_tokens":0,'
    r'"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"server_tool_use":{"web_search_requests":0}},'
    r'"modelUsage":{},"errors":["Internal error: \"Unauthorized (401) from '
    r'https://cli-chat-proxy.grok.com/v1/responses: Invalid or expired credentials (auth_kind=bearer, '
    r'x_xai_token_auth=xai-grok-cli, upstream=PermissionDenied, reason=no auth context)\\n\\n  Model:     '
    r'grok-4.6\\n  Auth:      ApiKey\\n  Version:   1.0.44\\n  Available: grok-4.6, grok-4.5\""],'
    r'"session_id":"00000000-0000-0000-0000-000000000005","uuid":"00000000-0000-0000-0000-000000000006"}' "\n"
)


class _GrokContainer(FakeCliContainer):
    """Also answers the read-backs of the staged agent definition and of
    the system prompt grok's session recorded: ``recorded``, by default
    what grok renders the definition's body to."""

    def __init__(self, *args, recorded=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = recorded

    def system_prompt(self) -> str:
        return self.copied["agent.md"].partition("\n---\n")[2].replace(
            '${{ "${" }}', "${").strip()

    def read_file(self, path, **kw):
        if path.endswith("/system_prompt.txt"):
            return self.system_prompt() if self.recorded is None else self.recorded
        if path.endswith("/agent.md"):
            return self.copied["agent.md"]
        return super().read_file(path, **kw)
_UNKNOWN_MODEL_RESULT = (
    r'{"type":"result","subtype":"error_during_execution","is_error":true,"duration_ms":0,"duration_api_ms":0,'
    r'"num_turns":0,"stop_reason":null,"total_cost_usd":0.0,"usage":{"input_tokens":0,"output_tokens":0,'
    r'"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"server_tool_use":{"web_search_requests":0}},'
    r'''"modelUsage":{},"errors":["Couldn't set model 'grok-9-nonexistent': Invalid params: '''
    r'''\"unknown model id\". Run 'grok models' to see available models."],'''
    r'"session_id":"00000000-0000-0000-0000-000000000007","uuid":"00000000-0000-0000-0000-000000000008"}' "\n"
)


def _grok_home(credential='{"scope": {"key": "k", "refresh_token": "r"}}') -> str:
    home = tempfile.mkdtemp(prefix="grok-home-")
    if credential is not None:
        Path(home, CREDENTIAL_NAME).write_text(credential, encoding="utf-8")
    return home


class BuildCommandTests(unittest.TestCase):
    def _cmd(self, **kwargs):
        defaults = dict(grok_bin="grok", model="grok-4.7", effort="high",
                        schema={"type": "object"}, agent_path="/run/agent.md")
        defaults.update(kwargs)
        return build_grok_command(**defaults)

    def test_the_flag_set(self) -> None:
        self.assertEqual(
            ["grok",
             "--agent", "/run/agent.md",
             "--single", "Do the task your system prompt gives you.",
             "--output-format", "streaming-messages-json",
             "--json-schema", '{"type": "object"}',
             "--model", "grok-4.7",
             "--tools", "search_tool,use_tool",
             "--deny", "Read(/**)",
             "--always-approve",
             "--effort", "high"],
            self._cmd())

    def test_effort_flag_only_when_set(self) -> None:
        self.assertNotIn("--effort", self._cmd(effort=None))


class ResultEventTests(unittest.TestCase):
    def test_the_result_line_is_found_behind_the_init_line(self) -> None:
        result = result_event(_REAL_STREAM)
        self.assertEqual("success", result["subtype"])
        self.assertEqual("reply_no_verdict", result["structured_output"]["classification"])

    def test_no_result_line(self) -> None:
        self.assertIsNone(result_event(_REAL_STREAM.splitlines()[0] + "\nnot json\n[1]\n"))


class GrokReviewerRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._relay_sessions = []

    def _reviewer(self, *, grok_home=None, **kw):
        return GrokReviewer("grok-4.7", name="grok:grok-4.7", role=ROLE,
                            grok_host=GROK_HOST,
                            grok_home=grok_home or _grok_home(), **kw)

    def _run(self, *, stdout=_REAL_STREAM, stderr="", returncode=0,
             ctx=None, reviewer=None, refreshed_credential=None, recorded=None):
        reviewer = reviewer or self._reviewer()
        proc = subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)
        self.container = _GrokContainer(
            proc, None, refreshed_auth=refreshed_credential, recorded=recorded)
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
        self.assertEqual("reply_no_verdict", result["classification"])

    def test_prompt_is_the_agent_definitions_system_prompt_only(self) -> None:
        self._run()
        self.assertTrue(self.container.copied["agent.md"].startswith(
            "---\nname: fairy\ndescription: one pass of fairy\npromptMode: full\n"
            "discoverSkills: false\nagentsMd: false\n---\n"))
        self.assertIn("the patch", self.container.system_prompt())
        self.assertIn("Review the following PR.", self.container.system_prompt())
        self.assertEqual("", self.container.input_text)
        self.assertNotIn("the patch", " ".join(self.container.cmd))
        self.assertIn("/work/.grok-run/agent.md", self.container.cmd)
        self.assertNotIn("output_schema.json", self.container.copied)
        self.assertEqual("/work/.grok-run", self.container.dest_dirs["agent.md"])

    def test_no_template_delimiter_of_the_pr_text_is_left_in_the_body(self) -> None:
        patch = "run: ${{ secrets.TOKEN }} ${% raw %} ${# x #} ${VAR} $${ {{ y }} %}"
        self._run(ctx=make_ctx(patch_text=patch))
        body = self.container.copied["agent.md"].partition("\n---\n")[2]
        self.assertNotIn("${", body.replace('${{ "${" }}', ""))
        self.assertIn(patch, self.container.system_prompt())

    def test_front_matter_lines_and_outer_whitespace_in_the_prompt(self) -> None:
        self._run(ctx=make_ctx(patch_text="\n---\nname: other\n---\n"),
                  recorded=None)
        self.assertIn("\n---\nname: other\n---\n", self.container.system_prompt())

    def test_a_run_on_another_system_prompt_is_an_error(self) -> None:
        for recorded in ("You are Grok.", ""):
            with self.subTest(recorded=recorded), \
                    self.assertRaisesRegex(RuntimeError, "staged system prompt"):
                self._run(recorded=recorded)

    def test_container_gets_credential_config_and_bridge(self) -> None:
        self._run()
        copied = self.container.copied
        for name in (CREDENTIAL_NAME, "config.toml", "mcp_shell_bridge.py",
                     "shell_bridge_client.py", "relay.py"):
            self.assertIn(name, copied)
        self.assertEqual(
            ["/grok-home", "/grok-home"],
            [self.container.dest_dirs[name] for name in (CREDENTIAL_NAME, "config.toml")])
        config = tomllib.loads(copied["config.toml"])
        self.assertEqual(
            {"command": "python3",
             "args": ["/work/.grok-run/mcp_shell_bridge.py",
                      "--socket", "/work/.grok-run/shell.sock",
                      "--machine", "x86_64", "--machine", "arm64"],
             "tool_timeout_sec": cli_reviewer.MCP_TOOL_TIMEOUT_S},
            config["mcp_servers"]["shell"])
        self.assertGreater(config["mcp"]["max_output_bytes"], 4 * 256 * 1024)

    def test_any_machine_label_survives_the_toml_config(self) -> None:
        label = 'a"b\\c\x7fd\né\U0001f600'
        machine = podman_host.parse_shell_host(f"{label}=fairy@h1")
        self._run(ctx=make_ctx(machines=(machine,)))
        config = tomllib.loads(self.container.copied["config.toml"])
        self.assertEqual(label, config["mcp_servers"]["shell"]["args"][-1])

    def test_container_env_is_the_private_home_without_update_telemetry_and_managed_mcps(self) -> None:
        self._run()
        self.assertEqual({"GROK_HOME": "/grok-home",
                          "GROK_DISABLE_AUTOUPDATER": "1",
                          "GROK_TELEMETRY_ENABLED": "false",
                          "GROK_MANAGED_MCPS_ENABLED": "false"}, self.container.env)

    def test_shellless_context_configures_no_mcp_server(self) -> None:
        self._run(ctx=make_ctx(open_shell=None))
        self.assertNotIn("config.toml", self.container.copied)
        self.assertIn("the patch", self.container.system_prompt())

    def test_cancelled_run_ends_the_turn(self) -> None:
        with self.assertRaises(GrokTurnFailed) as caught:
            self._run(stdout=_CANCELLED_RESULT)
        self.assertIsInstance(caught.exception, ProviderTurnFailed)
        self.assertIn("error_during_execution", str(caught.exception))
        self.assertIn("cancelled", str(caught.exception))
        self.assertEqual([], self._poisoned_after(
            GrokTurnFailed, stdout=_CANCELLED_RESULT))

    def test_a_run_that_fails_before_a_turn_without_credential_words_ends_the_turn(self) -> None:
        home = _grok_home()
        with self.assertRaisesRegex(GrokTurnFailed, "unknown model id"):
            self._run(reviewer=self._reviewer(grok_home=home),
                      stdout=_UNKNOWN_MODEL_RESULT, returncode=1)
        self.assertTrue(Path(home, CREDENTIAL_NAME).is_file())

    def test_missing_or_rejected_credential_sets_it_aside(self) -> None:
        home = _grok_home()
        for stdout in (_NOT_SIGNED_IN_RESULT, _REJECTED_TOKENS_RESULT):
            Path(home, CREDENTIAL_NAME).write_text("{}", encoding="utf-8")
            with self.subTest(stdout=stdout), self.assertRaises(CliAuthFailed), \
                    self.assertLogs(cli_reviewer.logger, "CRITICAL"):
                self._run(reviewer=self._reviewer(grok_home=home), stdout=stdout,
                          returncode=1)
            self.assertFalse(Path(home, CREDENTIAL_NAME).exists())
            self.assertTrue(Path(home, CREDENTIAL_NAME + ".invalid").is_file())

    def test_the_models_own_words_are_no_credential_failure(self) -> None:
        """The errors of a run that reached the model can hold the schema
        validator's message about its answer; this one is made up, grok
        having answered in schema whenever one was tried."""
        result = json.loads(_CANCELLED_RESULT)
        result["errors"] = ['output does not match the required schema: '
                            '"Not signed in" is not one of ["approve","skip"]']
        home = _grok_home()
        with self.assertRaises(GrokTurnFailed):
            self._run(reviewer=self._reviewer(grok_home=home),
                      stdout=json.dumps(result), returncode=1)
        self.assertTrue(Path(home, CREDENTIAL_NAME).is_file())

    def test_refreshed_credential_is_persisted_from_grok_home(self) -> None:
        home = _grok_home()
        rotated = '{"scope": {"key": "k2", "refresh_token": "rotated"}}'
        self._run(reviewer=self._reviewer(grok_home=home), refreshed_credential=rotated)
        self.assertEqual(rotated, Path(home, CREDENTIAL_NAME).read_text(encoding="utf-8"))
        self.assertTrue(Path(home, CREDENTIAL_NAME + ".bak").is_file())
        (read_back,) = [p for p, _ in self.container.read_file_calls]
        self.assertEqual(f"/grok-home/{CREDENTIAL_NAME}", read_back)

    def test_no_result_line_is_a_crash_that_poisons(self) -> None:
        poisoned = self._poisoned_after(RuntimeError, stdout="", returncode=137)
        self.assertEqual(1, len(poisoned))

    def test_debug_dump_is_prefixed_grok(self) -> None:
        debug_dir = tempfile.mkdtemp(prefix="grok-dbg-")
        self._run(reviewer=self._reviewer(debug_dir=debug_dir))
        (path,) = Path(debug_dir).iterdir()
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertRegex(payload["response"]["id"], r"^grok-\d{8}-\d{6}-review$")
        self.assertEqual(8166, payload["response"]["usage"]["input_tokens"])
        self.assertEqual(self.container.cmd, payload["request"]["cmd"])

    def test_missing_credential_is_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, f"{CREDENTIAL_NAME} not found"):
            self._run(reviewer=self._reviewer(grok_home=_grok_home(credential=None)))
        self.assertIsNone(self.container.cmd)

    def test_no_grok_host_is_error(self) -> None:
        reviewer = GrokReviewer("m", name="grok:m", role=ROLE, grok_home=_grok_home())
        with self.assertRaisesRegex(RuntimeError, "requires --grok-host"):
            reviewer.run(make_ctx())

    def test_invalid_effort_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GrokReviewer("m", name="grok:m", role=ROLE, effort="max")


class FactoryTests(unittest.TestCase):
    def _args(self, **extra):
        base = dict(
            grok_bin="grok-pinned", grok_timeout_seconds=0.0, grok_home=None,
            grok_host=GROK_HOST, grok_image="img:test",
            podman_exec_timeout=600.0, debug_response_dir=None,
        )
        base.update(extra)
        return argparse.Namespace(**base)

    def test_make_reviewer_builds_grok(self) -> None:
        reviewer = review_pipeline.make_reviewer(
            "grok:grok-4.7@xhigh", args=self._args(),
            resources=None, role=ROLE, verbose=False)
        self.assertIsInstance(reviewer, GrokReviewer)
        self.assertEqual("grok:grok-4.7", reviewer.name)
        self.assertEqual("xhigh", reviewer.effort)
        self.assertEqual("grok-pinned", reviewer.grok_bin)
        self.assertEqual(GROK_HOST, reviewer.host)
        self.assertEqual("img:test", reviewer.image)

    def test_grok_home_option_selects_the_login(self) -> None:
        reviewer = review_pipeline.make_reviewer(
            "grok:grok-4.7:grok-home=/second-home", args=self._args(),
            resources=None, role=ROLE, verbose=False)
        self.assertEqual(Path("/second-home", CREDENTIAL_NAME), reviewer.credential_local)
        self.assertEqual("grok:grok-4.7+second-home", reviewer.name)

    def test_grok_without_host_is_cli_error(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "grok:grok-4.7", args=self._args(grok_host=None),
                resources=None, role=ROLE, verbose=False)

    def test_bad_effort_suffix_is_cli_error(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "grok:grok-4.7@max", args=self._args(),
                resources=None, role=ROLE, verbose=False)


if __name__ == "__main__":
    unittest.main()
