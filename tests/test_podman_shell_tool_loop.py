"""Unit tests for podman-container shell tool helpers in the OpenAI wrapper."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import podman_host as lc  # noqa: E402
import openai_reviewer  # noqa: E402


class ExtractFunctionCallsTests(unittest.TestCase):
    def test_extracts_shell_calls_with_call_id(self) -> None:
        rsp = mock.Mock()
        rsp.id = "resp_1"
        with mock.patch.object(openai_reviewer, "response_to_debug_json", return_value={
            "output": [
                {"type": "message", "id": "m1"},
                {
                    "type": "function_call",
                    "call_id": "fc_abc",
                    "name": "shell",
                    "arguments": json.dumps({"command": "true"}),
                },
            ],
        }):
            calls = openai_reviewer.extract_function_calls_from_response(rsp)
        self.assertEqual(1, len(calls))
        self.assertEqual("shell", calls[0]["name"])
        self.assertEqual("fc_abc", calls[0]["call_id"])

    def test_falls_back_to_item_id(self) -> None:
        rsp = mock.Mock()
        with mock.patch.object(openai_reviewer, "response_to_debug_json", return_value={
            "output": [
                {
                    "type": "function_call",
                    "id": "item_xyz",
                    "name": "shell",
                    "arguments": "{}",
                },
            ],
        }):
            calls = openai_reviewer.extract_function_calls_from_response(rsp)
        self.assertEqual("item_xyz", calls[0]["call_id"])


class RunPodmanShellLoopTests(unittest.TestCase):
    def test_two_rounds_then_final(self) -> None:
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="hi\n", stderr="", duration_s=0.01,
            stdout_truncated=False, stderr_truncated=False,
        )
        r1 = mock.Mock()
        r1.id = "resp_1"
        r2 = mock.Mock()
        r2.id = "resp_2"

        dump_calls = {"c": 0}

        def dump(_rsp: object) -> dict:
            dump_calls["c"] += 1
            if dump_calls["c"] == 1:
                return {
                    "output": [{
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "shell",
                        "arguments": json.dumps({"command": "echo hi"}),
                    }],
                }
            return {"output": [{"type": "message", "content": []}]}

        creates: list[object] = []

        def fake_create(**kwargs: object) -> object:
            creates.append(kwargs)
            if len(creates) == 1:
                return r1
            return r2

        client = mock.Mock()
        client.responses.create.side_effect = fake_create

        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump):
            with mock.patch.object(openai_reviewer, "call_with_rate_limit_retry", side_effect=lambda fn, **kw: fn()):
                out = openai_reviewer.run_responses_resolving_podman_shell(
                    client,
                    initial_kwargs={
                        "model": "gpt-x",
                        "tools": [{"type": "function", "name": "shell"}],
                    },
                    shells={"x86_64": session},
                    machine_labels=("x86_64",),
                    open_shell=lambda l: (session, ""),
                    max_tool_rounds=10,
                    max_shell_timeout_s=60.0,
                    what="test",
                    verbose=False,
                )
        self.assertIs(r2, out)
        self.assertEqual(2, client.responses.create.call_count)
        session.exec.assert_called_once_with("echo hi", cwd=None, timeout_s=60.0)
        follow = creates[1]
        self.assertEqual("resp_1", follow["previous_response_id"])
        self.assertEqual(1, len(follow["input"]))
        self.assertEqual("function_call_output", follow["input"][0]["type"])
        self.assertEqual(False, follow.get("parallel_tool_calls"))

    def test_parallel_flag_lifts_serial_forcing(self) -> None:
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="", stderr="", duration_s=0.0,
            stdout_truncated=False, stderr_truncated=False,
        )
        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{
                    "type": "function_call", "call_id": "c1",
                    "name": "shell", "arguments": json.dumps({"command": "true"}),
                }]}
            return {"output": [{"type": "message", "content": []}]}

        creates: list[dict] = []
        client = mock.Mock()
        client.responses.create.side_effect = (
            lambda **kw: (creates.append(kw), mock.Mock(id=f"r{n['i']}"))[1]
        )
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump), \
                mock.patch.object(openai_reviewer, "call_with_rate_limit_retry",
                                  side_effect=lambda fn, **kw: fn()):
            openai_reviewer.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={
                    "model": "gpt-x",
                    "tools": [{"type": "function", "name": "shell"}],
                },
                podman_shell_session=session,
                max_tool_rounds=10,
                max_shell_timeout_s=60.0,
                what="test",
                verbose=False,
                parallel_tool_calls=True,
            )
        self.assertNotIn("parallel_tool_calls", creates[1])

    def test_every_round_is_dumped_with_its_own_kwargs(self) -> None:
        # Regression: only the final response used to be dumped, losing the
        # intermediate rounds where the function calls and outputs live
        # (observed on the 2026-07-02 ensemble run: the combiner's shell
        # call was visible on the OpenAI console but absent from openaidebug).
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="", stderr="", duration_s=0.0,
            stdout_truncated=False, stderr_truncated=False,
        )
        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{
                    "type": "function_call", "call_id": "c1",
                    "name": "shell", "arguments": json.dumps({"command": "true"}),
                }]}
            return {"output": [{"type": "message", "content": []}]}

        client = mock.Mock()
        client.responses.create.side_effect = lambda **kw: mock.Mock(id=f"r{n['i']}")
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump), \
                mock.patch.object(openai_reviewer, "call_with_rate_limit_retry",
                                  side_effect=lambda fn, **kw: fn()), \
                mock.patch.object(openai_reviewer, "dump_response_debug_artifacts") as dumped:
            openai_reviewer.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={"model": "gpt-x", "tools": [{"type": "function", "name": "shell"}]},
                shells={"x86_64": session},
                machine_labels=("x86_64",),
                open_shell=lambda l: (session, ""),
                max_tool_rounds=10,
                max_shell_timeout_s=60.0,
                what="test",
                verbose=False,
                debug_dir="/tmp/dbg",
                wrapper_request={"pull_request": {"number": 7}},
            )
        self.assertEqual(2, dumped.call_count)
        first_kwargs = dumped.call_args_list[0].args[1]
        follow_kwargs = dumped.call_args_list[1].args[1]
        self.assertNotIn("previous_response_id", first_kwargs)
        self.assertEqual("function_call_output", follow_kwargs["input"][0]["type"])
        for call in dumped.call_args_list:
            self.assertEqual("/tmp/dbg", call.kwargs["debug_dir"])
            self.assertEqual({"pull_request": {"number": 7}}, call.kwargs["wrapper_request"])
        # Follow-up rounds append to the file the first round created.
        self.assertIsNone(dumped.call_args_list[0].kwargs["conversation"])
        self.assertIs(dumped.return_value,
                      dumped.call_args_list[1].kwargs["conversation"])

    def test_follow_up_forwards_text_format(self) -> None:
        # Regression: the json_schema output format is per-request and not
        # inherited via previous_response_id. Without forwarding it, a
        # review that ends after shell rounds could answer off-schema
        # (observed 2026-07-03, ab-gptonly HEAD_3 PR 20997: three straight
        # attempts returned {"class": ...} instead of {"classification":
        # ...} and the review failed).
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="", stderr="", duration_s=0.0,
            stdout_truncated=False, stderr_truncated=False,
        )
        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{
                    "type": "function_call", "call_id": "c1",
                    "name": "shell", "arguments": json.dumps({"command": "true"}),
                }]}
            return {"output": [{"type": "message", "content": []}]}

        text_format = {"format": {"type": "json_schema", "name": "review_verdict"}}
        creates: list[dict] = []
        client = mock.Mock()
        client.responses.create.side_effect = (
            lambda **kw: (creates.append(kw), mock.Mock(id=f"r{n['i']}"))[1]
        )
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump), \
                mock.patch.object(openai_reviewer, "call_with_rate_limit_retry",
                                  side_effect=lambda fn, **kw: fn()):
            openai_reviewer.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={
                    "model": "gpt-x",
                    "tools": [{"type": "function", "name": "shell"}],
                    "text": text_format,
                },
                shells={"x86_64": session},
                machine_labels=("x86_64",),
                open_shell=lambda l: (session, ""),
                max_tool_rounds=10,
                max_shell_timeout_s=60.0,
                what="test",
                verbose=False,
            )
        self.assertEqual(2, len(creates))
        self.assertEqual(text_format, creates[1]["text"])

    def test_follow_up_forwards_reasoning(self) -> None:
        # Regression: gpt-5.6 keys its prompt cache on ``reasoning``; without
        # it every round 2 was a full-prefix miss (2026-07-11..13, 134/134)
        # and ran the round without the requested effort.
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="", stderr="", duration_s=0.0,
            stdout_truncated=False, stderr_truncated=False,
        )
        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{
                    "type": "function_call", "call_id": "c1",
                    "name": "shell", "arguments": json.dumps({"command": "true"}),
                }]}
            return {"output": [{"type": "message", "content": []}]}

        reasoning = {"effort": "high", "summary": "detailed"}
        creates: list[dict] = []
        client = mock.Mock()
        client.responses.create.side_effect = (
            lambda **kw: (creates.append(kw), mock.Mock(id=f"r{n['i']}"))[1]
        )
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump), \
                mock.patch.object(openai_reviewer, "call_with_rate_limit_retry",
                                  side_effect=lambda fn, **kw: fn()):
            openai_reviewer.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={
                    "model": "gpt-x",
                    "tools": [{"type": "function", "name": "shell"}],
                    "reasoning": reasoning,
                },
                shells={"x86_64": session},
                machine_labels=("x86_64",),
                open_shell=lambda l: (session, ""),
                max_tool_rounds=10,
                max_shell_timeout_s=60.0,
                what="test",
                verbose=False,
            )
        self.assertEqual(2, len(creates))
        self.assertEqual(reasoning, creates[1]["reasoning"])

    def test_follow_up_forwards_prompt_cache_key(self) -> None:
        # The key exists to route all rounds to one cache machine, so the
        # loop must forward it.
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="", stderr="", duration_s=0.0,
            stdout_truncated=False, stderr_truncated=False,
        )
        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{
                    "type": "function_call", "call_id": "c1",
                    "name": "shell", "arguments": json.dumps({"command": "true"}),
                }]}
            return {"output": [{"type": "message", "content": []}]}

        pck = "fairy:reviewer:22290:32486a55dc96"
        creates: list[dict] = []
        client = mock.Mock()
        client.responses.create.side_effect = (
            lambda **kw: (creates.append(kw), mock.Mock(id=f"r{n['i']}"))[1]
        )
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump), \
                mock.patch.object(openai_reviewer, "call_with_rate_limit_retry",
                                  side_effect=lambda fn, **kw: fn()):
            openai_reviewer.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={
                    "model": "gpt-x",
                    "tools": [{"type": "function", "name": "shell"}],
                    "prompt_cache_key": pck,
                },
                podman_shell_session=session,
                max_tool_rounds=10,
                max_shell_timeout_s=60.0,
                what="test",
                verbose=False,
            )
        self.assertEqual(2, len(creates))
        self.assertEqual(pck, creates[1]["prompt_cache_key"])

    def test_max_tool_rounds_zero_means_unlimited(self) -> None:
        session = mock.Mock(spec=lc.ContainerShellSession)
        session.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="", stderr="", duration_s=0.0,
            stdout_truncated=False, stderr_truncated=False,
        )
        # Always returns a pending shell call until we stop it, so a finite
        # cap would raise; with rounds=0 it must keep going.
        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] >= 25:
                return {"output": [{"type": "message", "content": []}]}
            return {"output": [{
                "type": "function_call", "call_id": f"c{n['i']}",
                "name": "shell", "arguments": json.dumps({"command": "true"}),
            }]}

        client = mock.Mock()
        client.responses.create.side_effect = lambda **kw: mock.Mock(id="r")
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump):
            with mock.patch.object(openai_reviewer, "call_with_rate_limit_retry", side_effect=lambda fn, **kw: fn()):
                openai_reviewer.run_responses_resolving_podman_shell(
                    client,
                    initial_kwargs={"model": "gpt-x", "tools": [{"type": "function", "name": "shell"}]},
                    shells={"x86_64": session},
                    machine_labels=("x86_64",),
                    open_shell=lambda l: (session, ""),
                    max_tool_rounds=0,
                    max_shell_timeout_s=60.0,
                    what="test",
                    verbose=False,
                )
        self.assertGreaterEqual(session.exec.call_count, 24)

    def test_machine_arg_routes_to_lazy_second_session(self) -> None:
        # only the default machine is opened eagerly
        default = mock.Mock(spec=lc.ContainerShellSession)
        arm = mock.Mock(spec=lc.ContainerShellSession)
        arm.exec.return_value = lc.ExecResult(
            exit_code=0, stdout="aarch64\n", stderr="", duration_s=0.01,
            stdout_truncated=False, stderr_truncated=False,
        )
        opened: list[str] = []

        def open_shell(label: str) -> tuple[object, str]:
            opened.append(label)
            return arm, "$ git status --short\nclean\n"

        n = {"i": 0}

        def dump(_rsp: object) -> dict:
            n["i"] += 1
            if n["i"] == 1:
                return {"output": [{
                    "type": "function_call", "call_id": "c1", "name": "shell",
                    "arguments": json.dumps(
                        {"command": "uname -m", "machine": "arm64"}),
                }]}
            return {"output": [{"type": "message", "content": []}]}

        creates: list[dict] = []

        def fake_create(**kwargs: object) -> object:
            creates.append(kwargs)
            return mock.Mock(id=f"r{len(creates)}")

        client = mock.Mock()
        client.responses.create.side_effect = fake_create
        with mock.patch.object(openai_reviewer, "response_to_debug_json", side_effect=dump):
            with mock.patch.object(openai_reviewer, "call_with_rate_limit_retry", side_effect=lambda fn, **kw: fn()):
                openai_reviewer.run_responses_resolving_podman_shell(
                    client,
                    initial_kwargs={"model": "gpt-x", "tools": []},
                    shells={"x86_64": default},
                    machine_labels=("x86_64", "arm64"),
                    open_shell=open_shell,
                    max_tool_rounds=10,
                    max_shell_timeout_s=60.0,
                    what="test",
                    verbose=False,
                )
        self.assertEqual(["arm64"], opened)
        default.exec.assert_not_called()
        arm.exec.assert_called_once_with("uname -m", cwd=None, timeout_s=60.0)
        payload = json.loads(creates[1]["input"][0]["output"])
        self.assertEqual("aarch64\n", payload["stdout"])
        self.assertEqual("$ git status --short\nclean\n", payload["setup_transcript"])


class ShellToolSchemaTests(unittest.TestCase):
    def _spec(self, label: str) -> lc.ShellHostSpec:
        return lc.ShellHostSpec(label, lc.RemoteHost("fairy@h"))

    def test_single_machine_schema_has_no_machine_parameter(self) -> None:
        tool = openai_reviewer.build_podman_shell_function_tool([self._spec("x86_64")])
        self.assertNotIn("machine", tool["parameters"]["properties"])
        self.assertEqual(["command"], tool["parameters"]["required"])

    def test_two_machines_expose_enum(self) -> None:
        tool = openai_reviewer.build_podman_shell_function_tool(
            [self._spec("x86_64"), self._spec("arm64")])
        machine = tool["parameters"]["properties"]["machine"]
        self.assertEqual(["x86_64", "arm64"], machine["enum"])
        self.assertIn("default x86_64", machine["description"])
        self.assertNotIn("machine", tool["parameters"]["required"])

