"""Unit tests for podman-container shell tool helpers in the OpenAI wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import podman_host as lc  # noqa: E402
import openai_pr_review_wrapper as wrapper  # noqa: E402


class BuildRemoteHostTests(unittest.TestCase):
    def test_builds_host_from_ssh_dest(self) -> None:
        args = argparse.Namespace(podman_ssh_dest="fairy@h", podman_ssh_identity=None)
        host = wrapper._build_remote_host(args)
        self.assertEqual("fairy@h", host.ssh_dest)
        self.assertIsNone(host.identity)

    def test_passes_identity_when_set(self) -> None:
        args = argparse.Namespace(podman_ssh_dest="fairy@h", podman_ssh_identity="/k/id")
        self.assertEqual("/k/id", wrapper._build_remote_host(args).identity)


class ExtractFunctionCallsTests(unittest.TestCase):
    def test_extracts_shell_calls_with_call_id(self) -> None:
        rsp = mock.Mock()
        rsp.id = "resp_1"
        with mock.patch.object(wrapper, "response_to_debug_json", return_value={
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
            calls = wrapper.extract_function_calls_from_response(rsp)
        self.assertEqual(1, len(calls))
        self.assertEqual("shell", calls[0]["name"])
        self.assertEqual("fc_abc", calls[0]["call_id"])

    def test_falls_back_to_item_id(self) -> None:
        rsp = mock.Mock()
        with mock.patch.object(wrapper, "response_to_debug_json", return_value={
            "output": [
                {
                    "type": "function_call",
                    "id": "item_xyz",
                    "name": "shell",
                    "arguments": "{}",
                },
            ],
        }):
            calls = wrapper.extract_function_calls_from_response(rsp)
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

        with mock.patch.object(wrapper, "response_to_debug_json", side_effect=dump):
            with mock.patch.object(wrapper, "call_with_rate_limit_retry", side_effect=lambda fn, **kw: fn()):
                out = wrapper.run_responses_resolving_podman_shell(
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
                )
        self.assertIs(r2, out)
        self.assertEqual(2, client.responses.create.call_count)
        session.exec.assert_called_once_with("echo hi", cwd=None, timeout_s=60.0)
        follow = creates[1]
        self.assertEqual("resp_1", follow["previous_response_id"])
        self.assertEqual(1, len(follow["input"]))
        self.assertEqual("function_call_output", follow["input"][0]["type"])

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
        with mock.patch.object(wrapper, "response_to_debug_json", side_effect=dump), \
                mock.patch.object(wrapper, "call_with_rate_limit_retry",
                                  side_effect=lambda fn, **kw: fn()), \
                mock.patch.object(wrapper, "dump_response_debug_artifacts") as dumped:
            wrapper.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={"model": "gpt-x", "tools": [{"type": "function", "name": "shell"}]},
                podman_shell_session=session,
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
        with mock.patch.object(wrapper, "response_to_debug_json", side_effect=dump):
            with mock.patch.object(wrapper, "call_with_rate_limit_retry", side_effect=lambda fn, **kw: fn()):
                wrapper.run_responses_resolving_podman_shell(
                    client,
                    initial_kwargs={"model": "gpt-x", "tools": [{"type": "function", "name": "shell"}]},
                    podman_shell_session=session,
                    max_tool_rounds=0,
                    max_shell_timeout_s=60.0,
                    what="test",
                    verbose=False,
                )
        self.assertGreaterEqual(session.exec.call_count, 24)

