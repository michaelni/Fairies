"""CodexReviewer: command construction, event parsing, run() behavior."""

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codex_container
import codex_reviewer
import podman_host
import review_pipeline
from codex_reviewer import (
    CodexReviewer,
    CodexUsageLimit,
    build_codex_exec_command,
    resolve_web_search,
    summarize_codex_events,
)
from llm_review_api import BadModelOutput, ReviewContext, RoleSpec

MACHINES = (
    podman_host.parse_shell_host("fairy@h1"),
    podman_host.parse_shell_host("arm64=fairy@h2"),
)

CODEX_HOST = podman_host.parse_shell_host("fairy@codexbox")


def _codex_home_with_auth(**files) -> str:
    """A temp CODEX_HOME containing auth.json (+ optional extra files)."""
    home = tempfile.mkdtemp(prefix="codex-home-")
    Path(home, "auth.json").write_text('{"tokens": {}}', encoding="utf-8")
    for name, content in files.items():
        Path(home, name).write_text(content, encoding="utf-8")
    return home


class _FakeCodexContainer:
    """Stand-in for CodexContainer: records podman-cp'd files and the codex
    argv, returns canned run output / last message. ``on_run`` fires inside
    run() (while the auth flock is held) for lock tests."""

    def __init__(self, run_result, last_message, on_run=None,
                 refreshed_auth=None):
        self._run_result = run_result
        self._last_message = last_message
        self.on_run = on_run
        # harness knob: read_file(auth.json) returns this instead of the copied auth
        self._refreshed_auth = refreshed_auth
        self.copied = {}       # basename -> local text content
        self.cmd = None
        self.input_text = None
        self.env = None
        self.stopped = False

    def start(self):
        return self

    def put_file(self, local, dest_dir, **kw):
        p = Path(local)
        try:
            self.copied[p.name] = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self.copied[p.name] = None

    def run(self, cmd, *, input_text=None, env=None, timeout_s=None):
        self.cmd = cmd
        self.input_text = input_text
        self.env = env
        if self.on_run is not None:
            self.on_run()
        return self._run_result

    def read_file(self, path, **kw):
        if path.endswith("auth.json"):
            if self._refreshed_auth is not None:
                return self._refreshed_auth
            return self.copied.get("auth.json")
        return self._last_message

    def stop(self):
        self.stopped = True

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
        machines=MACHINES,
        open_shell=lambda label: (
            mock.Mock(spec=podman_host.ContainerShellSession), ""),
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
        self.assertIn("--sandbox danger-full-access", joined)
        self.assertIn("features.shell_tool=false", cmd)
        self.assertIn("features.unified_exec=false", cmd)
        self.assertIn("analytics.enabled=false", cmd)
        self.assertIn('web_search="cached"', cmd)
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)
        self.assertNotIn("--yolo", cmd)

    def test_web_search_mode_is_emitted(self) -> None:
        self.assertIn('web_search="live"',
                      self._cmd(web_search="live"))
        self.assertIn('web_search="disabled"',
                      self._cmd(web_search="disabled"))

    def test_verbosity_and_reasoning_summary_emitted(self) -> None:
        cmd = self._cmd(verbosity="high", reasoning_summary="detailed")
        self.assertIn('model_verbosity="high"', cmd)
        self.assertIn('model_reasoning_summary="detailed"', cmd)
        bare = self._cmd()
        self.assertFalse(any("model_verbosity" in c for c in bare))
        self.assertFalse(any("model_reasoning_summary" in c for c in bare))

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

    def test_unified_exec_pinned_off(self) -> None:
        self.assertIn("features.unified_exec=false", self._cmd())

    def test_catalog_override_flag_only_when_path_given(self) -> None:
        self.assertNotIn("model_catalog_json", " ".join(self._cmd()))
        cmd = self._cmd(catalog_override_path="/tmp/s/hardened_catalog.json")
        self.assertIn("model_catalog_json=/tmp/s/hardened_catalog.json", cmd)


class WebSearchModeTests(unittest.TestCase):
    def test_resolve_maps_flag_to_mode(self) -> None:
        self.assertEqual("disabled", resolve_web_search("off"))
        self.assertEqual("live", resolve_web_search("live"))
        self.assertEqual("cached", resolve_web_search("cached"))

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CodexReviewer("m", name="codex:m", role=ROLE, codex_host=CODEX_HOST,
                          web_search="sometimes")

    def _prompt_features(self, web_search):
        reviewer = CodexReviewer("m", name="codex:m", role=ROLE,
                                 codex_host=CODEX_HOST, web_search=web_search)
        with mock.patch.object(codex_reviewer, "generate_llm_prompt",
                               return_value="DEV") as gen:
            reviewer._build_prompt(_ctx(), False)
        return gen.call_args.kwargs["features"]

    def test_web_search_feature_tracks_mode(self) -> None:
        self.assertNotIn("web_search", self._prompt_features("disabled"))
        self.assertIn("web_search", self._prompt_features("live"))
        self.assertIn("web_search", self._prompt_features("cached"))


class HardenCatalogTests(unittest.TestCase):
    CATALOG = {
        "etag": "abc",
        "models": [
            {"slug": "gpt-5.6-sol", "input_modalities": ["text", "image"],
             "apply_patch_tool_type": "freeform", "tool_mode": "code_mode_only",
             "context_window": 400000},
            {"slug": "gpt-5.5", "input_modalities": ["text", "image"],
             "apply_patch_tool_type": "freeform", "tool_mode": None},
        ],
    }

    def test_strips_direct_host_file_tools(self) -> None:
        out = codex_reviewer.harden_codex_catalog(self.CATALOG)
        for m in out["models"]:
            self.assertEqual(["text"], m["input_modalities"])  # view_image inert
            self.assertIsNone(m["apply_patch_tool_type"])      # no apply_patch
        # unrelated fields survive
        self.assertEqual(400000, out["models"][0]["context_window"])
        self.assertEqual("abc", out["etag"])

    def test_tool_mode_left_untouched(self) -> None:
        out = codex_reviewer.harden_codex_catalog(self.CATALOG)
        self.assertEqual("code_mode_only", out["models"][0]["tool_mode"])
        self.assertIsNone(out["models"][1]["tool_mode"])

    def test_does_not_mutate_input(self) -> None:
        codex_reviewer.harden_codex_catalog(self.CATALOG)
        first = self.CATALOG["models"][0]
        self.assertEqual(["text", "image"], first["input_modalities"])
        self.assertEqual("freeform", first["apply_patch_tool_type"])


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
    def _reviewer(self, *, codex_home=None, **kw):
        return CodexReviewer(
            "gpt-5.6-sol", name="codex:gpt-5.6-sol", role=ROLE,
            codex_host=CODEX_HOST,
            codex_home=codex_home or _codex_home_with_auth(), **kw)

    def _run(self, *, jsonl="", stderr="", returncode=0,
             last_message='{"classification": "approve", "message": "ok"}',
             ctx=None, on_run=None, reviewer=None, refreshed_auth=None):
        reviewer = reviewer or self._reviewer()
        proc = subprocess.CompletedProcess(
            [], returncode, stdout=jsonl, stderr=stderr)
        self.container = _FakeCodexContainer(
            proc, last_message, on_run=on_run, refreshed_auth=refreshed_auth)
        with mock.patch.object(codex_reviewer, "CodexContainer",
                               return_value=self.container), \
                mock.patch.object(codex_reviewer, "CodexShellRelay") as relay:
            relay.return_value.start.return_value = relay.return_value
            relay.return_value.opened_sessions.return_value = \
                getattr(self, "_relay_sessions", [])
            return reviewer.run(ctx or _ctx())

    def test_validated_result_from_last_message(self) -> None:
        result = self._run()
        self.assertEqual("approve", result["classification"])
        self.assertIn("the patch", self.container.input_text)
        self.assertIn("Review the following PR.", self.container.input_text)

    def test_container_gets_auth_and_bridge_files(self) -> None:
        self._run()
        for name in ("auth.json", "codex_bridge.py", "shell_bridge_client.py",
                     "relay.py", "output_schema.json"):
            self.assertIn(name, self.container.copied)

    def test_only_codex_home_reaches_container_env(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-x",
                                          "CODEX_API_KEY": "sk-y"}):
            self._run()
        self.assertEqual(
            {"CODEX_HOME": codex_container.CONTAINER_CODEX_HOME},
            self.container.env,
        )

    def test_hardened_catalog_copied_and_passed(self) -> None:
        home = _codex_home_with_auth(**{"models_cache.json": json.dumps({
            "models": [{"slug": "gpt-5.6-sol",
                        "input_modalities": ["text", "image"],
                        "apply_patch_tool_type": "freeform",
                        "tool_mode": "code_mode_only"}]})})
        self._run(reviewer=self._reviewer(codex_home=home))
        self.assertIn("hardened_catalog.json", self.container.copied)
        entry = json.loads(
            self.container.copied["hardened_catalog.json"])["models"][0]
        self.assertEqual(["text"], entry["input_modalities"])
        self.assertIsNone(entry["apply_patch_tool_type"])
        self.assertEqual("code_mode_only", entry["tool_mode"])
        self.assertTrue(any(c.startswith("model_catalog_json=")
                            for c in self.container.cmd))

    def test_missing_catalog_skips_override(self) -> None:
        self._run()  # auth only, no models_cache.json
        self.assertNotIn("hardened_catalog.json", self.container.copied)
        self.assertFalse(
            any("model_catalog_json" in c for c in self.container.cmd))

    def test_usage_limit_is_hard_failure(self) -> None:
        jsonl = json.dumps({"type": "turn.failed", "error": {
            "message": "You've hit your usage limit",
            "type": "usage_limit_reached"}})
        with self.assertRaises(CodexUsageLimit):
            self._run(jsonl=jsonl, last_message=None, returncode=1)

    def test_missing_final_message_is_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no final message"):
            self._run(last_message=None)

    def test_non_json_final_message_is_bad_model_output(self) -> None:
        with self.assertRaises(BadModelOutput):
            self._run(last_message="I approve of this patch.")

    def test_refreshed_auth_persisted_back(self) -> None:
        home = _codex_home_with_auth()
        reviewer = self._reviewer(codex_home=home)
        rotated = '{"tokens": {"refresh_token": "rotated"}}'
        self._run(reviewer=reviewer, refreshed_auth=rotated)
        auth_path = Path(home, "auth.json")
        self.assertEqual(rotated, auth_path.read_text(encoding="utf-8"))
        self.assertEqual(0o600, auth_path.stat().st_mode & 0o777)

    def test_unchanged_auth_not_rewritten(self) -> None:
        home = _codex_home_with_auth()
        auth_path = Path(home, "auth.json")
        before = auth_path.read_text(encoding="utf-8")
        # refreshed_auth defaults to the copied auth (identical) -> no write.
        self._run(reviewer=self._reviewer(codex_home=home))
        self.assertEqual(before, auth_path.read_text(encoding="utf-8"))

    def test_crash_poisons_review_containers(self) -> None:
        s1 = mock.Mock(spec=podman_host.ContainerShellSession)
        s2 = mock.Mock(spec=podman_host.ContainerShellSession)
        self._relay_sessions = [s1, s2]
        self.addCleanup(lambda: delattr(self, "_relay_sessions"))
        poisoned = []
        ctx = _ctx(report_poisoned=poisoned.append)
        with self.assertRaises(RuntimeError):
            self._run(ctx=ctx, last_message=None)
        self.assertEqual([s1, s2], poisoned)

    def test_bad_output_does_not_poison(self) -> None:
        self._relay_sessions = [mock.Mock(spec=podman_host.ContainerShellSession)]
        self.addCleanup(lambda: delattr(self, "_relay_sessions"))
        poisoned = []
        ctx = _ctx(report_poisoned=poisoned.append)
        with self.assertRaises(BadModelOutput):
            self._run(ctx=ctx, last_message="not json")
        self.assertEqual([], poisoned)  # codex ran fine; not suspect

    def test_usage_limit_does_not_poison(self) -> None:
        self._relay_sessions = [mock.Mock(spec=podman_host.ContainerShellSession)]
        self.addCleanup(lambda: delattr(self, "_relay_sessions"))
        poisoned = []
        ctx = _ctx(report_poisoned=poisoned.append)
        jsonl = json.dumps({"type": "turn.failed",
                            "error": {"type": "usage_limit_reached"}})
        with self.assertRaises(CodexUsageLimit):
            self._run(ctx=ctx, jsonl=jsonl, last_message=None)
        self.assertEqual([], poisoned)  # clean quota stop

    def test_shellless_context_omits_mcp(self) -> None:
        self._run(ctx=_ctx(open_shell=None))
        self.assertFalse(any("mcp_servers" in c for c in self.container.cmd))

    def test_no_codex_host_is_error(self) -> None:
        reviewer = CodexReviewer("gpt-5.6-sol", name="codex:gpt-5.6-sol",
                                 role=ROLE, codex_home=_codex_home_with_auth())
        with self.assertRaisesRegex(RuntimeError, "requires --codex-host"):
            reviewer.run(_ctx())

    def test_missing_auth_is_error(self) -> None:
        home = tempfile.mkdtemp(prefix="codex-home-noauth-")  # no auth.json
        with self.assertRaisesRegex(RuntimeError, "auth.json not found"):
            self._run(reviewer=self._reviewer(codex_home=home))

    def test_invalid_effort_rejected(self) -> None:
        # "minimal" is codex's documented lowest effort but the server
        # rejects it (see CODEX_EFFORTS); it must not validate here.
        with self.assertRaises(ValueError):
            CodexReviewer("m", name="codex:m", role=ROLE, effort="minimal")

    def test_passes_are_serialized(self) -> None:
        # One auth.json must not serve concurrent jobs.
        import threading
        import time
        active = []
        reviewer = self._reviewer(codex_home=_codex_home_with_auth())

        def make_fake(*a, **k):
            def busy():
                active.append(1)
                self.assertEqual(1, len(active))
                time.sleep(0.02)
                active.pop()
            return _FakeCodexContainer(
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                '{"classification": "approve", "message": "ok"}', on_run=busy)

        with mock.patch.object(codex_reviewer, "CodexContainer",
                               side_effect=make_fake), \
                mock.patch.object(codex_reviewer, "CodexShellRelay") as relay:
            relay.return_value.start.return_value = relay.return_value
            threads = [threading.Thread(target=reviewer.run, args=(_ctx(),))
                       for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()


class FactoryTests(unittest.TestCase):
    def _args(self, **extra):
        base = dict(
            codex_bin="codex-pinned", codex_timeout_seconds=0.0,
            codex_home=None, codex_host=CODEX_HOST, codex_image="img:test",
            podman_exec_timeout=600.0, debug_response_dir=None,
            podman_max_tool_rounds=0,
            web_search="off", verbosity="high", reasoning_summary="detailed",
        )
        base.update(extra)
        return argparse.Namespace(**base)

    def test_make_reviewer_builds_codex(self) -> None:
        reviewer = review_pipeline.make_reviewer(
            "codex:gpt-5.6-sol@xhigh", args=self._args(), resources=None,
            role=ROLE, verbose=False,
        )
        self.assertIsInstance(reviewer, CodexReviewer)
        self.assertEqual("codex:gpt-5.6-sol", reviewer.name)
        self.assertEqual("xhigh", reviewer.effort)
        self.assertEqual("codex-pinned", reviewer.codex_bin)
        self.assertEqual(CODEX_HOST, reviewer.codex_host)
        self.assertEqual("disabled", reviewer.web_search)  # --web-search off

    def test_make_reviewer_web_search_modes(self) -> None:
        live = review_pipeline.make_reviewer(
            "codex:gpt-5.6-sol", args=self._args(web_search="live"),
            resources=None, role=ROLE, verbose=False)
        self.assertEqual("live", live.web_search)
        cached = review_pipeline.make_reviewer(
            "codex:gpt-5.6-sol", args=self._args(web_search="cached"),
            resources=None, role=ROLE, verbose=False)
        self.assertEqual("cached", cached.web_search)

    def test_codex_without_host_is_cli_error(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "codex:gpt-5.6-sol", args=self._args(codex_host=None),
                resources=None, role=ROLE, verbose=False,
            )

    def test_bad_effort_suffix_is_cli_error(self) -> None:
        with self.assertRaises(SystemExit):
            review_pipeline.make_reviewer(
                "codex:gpt-5.6-sol@minimal", args=self._args(), resources=None,
                role=ROLE, verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
