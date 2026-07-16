"""--session-command: pre-run shell commands spliced into the prompt."""
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import pr_review_wrapper
import shell_tool
from llm_prompt import (
    COMBINER_ROLE,
    ISSUE_COMBINER_ROLE,
    ISSUE_INVESTIGATOR_ROLE,
    REVIEWER_ROLE,
    make_session_transcript_texts,
)
from llm_review_api import ReviewContext


def _result(rc=0, out="", err="", trunc=False):
    return podman_host.ExecResult(
        exit_code=rc, stdout=out, stderr=err, duration_s=0.01,
        stdout_truncated=trunc, stderr_truncated=False,
    )


def _ctx(transcript=""):
    return ReviewContext(
        request={"pull_request": {"number": 7, "title": "t"}},
        patch_text="p", patch_truncated=False, source_bundle=None,
        source_files=[], source_notes=[], reviewer_username="fairy",
        ci_triage_mode=False, repo_roots=[], repo_mount_paths=[],
        session_transcript=transcript,
    )


class RunSessionCommandsTests(unittest.TestCase):
    def test_transcript_shows_commands_outputs_and_failures(self) -> None:
        session = mock.Mock(spec=podman_host.ContainerShellSession)
        session.exec.side_effect = [
            _result(out="M file.c\n"),
            _result(rc=128, err="fatal: nope\n"),
            _result(out="big\n", trunc=True),
        ]
        t = shell_tool.run_session_commands(
            session, ["git status --short", "git bad", "git log"],
            max_timeout_s=60.0,
        )
        self.assertIn("$ git status --short\nM file.c\n", t)
        self.assertIn("$ git bad\nfatal: nope\n(exit 128)\n", t)
        self.assertIn("$ git log\nbig\n(output truncated)\n", t)

    def test_session_commands_get_full_timeout_budget(self) -> None:
        # Operator commands (e.g. a future git fetch) must not be capped at
        # the model-call default, but at the operator's exec timeout.
        session = mock.Mock(spec=podman_host.ContainerShellSession)
        session.exec.return_value = _result()
        shell_tool.run_session_commands(session, ["true"], max_timeout_s=600.0)
        self.assertEqual(600.0, session.exec.call_args.kwargs["timeout_s"])


class ExecMachineCallTests(unittest.TestCase):
    def _sessions(self):
        sessions = {}
        for label in ("x86_64", "arm64"):
            s = mock.Mock(spec=podman_host.ContainerShellSession)
            s.exec.return_value = _result(out=f"on {label}\n")
            sessions[label] = s
        return sessions

    def _call(self, shells, opened, args, labels=("x86_64", "arm64"),
              transcripts=None):
        sessions = self._sessions()

        def open_shell(label):
            opened.append(label)
            return sessions[label], (transcripts or {}).get(label, "")

        return shell_tool.exec_machine_call(
            shells, labels, open_shell, args, max_timeout_s=60.0,
        )

    def test_lazy_open_only_on_use(self) -> None:
        shells, opened = {}, []
        self._call(shells, opened, {"command": "true"})
        self.assertEqual(["x86_64"], opened)
        self.assertEqual(["x86_64"], list(shells))
        self._call(shells, opened, {"command": "true"})
        self.assertEqual(["x86_64"], opened)  # reused, not reopened

    def test_machine_arg_routes_to_named_session(self) -> None:
        shells, opened = {}, []
        payload = self._call(shells, opened, {"command": "true", "machine": "arm64"})
        self.assertEqual(["arm64"], opened)
        self.assertEqual("on arm64\n", payload["stdout"])

    def test_unknown_machine_errors_without_open(self) -> None:
        shells, opened = {}, []
        payload = self._call(shells, opened, {"command": "true", "machine": "riscv"})
        self.assertEqual(
            "unknown machine 'riscv'; available: x86_64, arm64", payload["error"])
        self.assertEqual([], opened)

    def test_setup_transcript_only_on_first_non_default_result(self) -> None:
        shells, opened = {}, []
        transcripts = {"x86_64": "$ true\n", "arm64": "$ git status\nclean\n"}
        first = self._call(shells, opened, {"command": "a", "machine": "arm64"},
                           transcripts=transcripts)
        self.assertEqual("$ git status\nclean\n", first["setup_transcript"])
        second = self._call(shells, opened, {"command": "b", "machine": "arm64"},
                            transcripts=transcripts)
        self.assertNotIn("setup_transcript", second)

    def test_default_machine_never_gets_setup_transcript(self) -> None:
        shells, opened = {}, []
        payload = self._call(shells, opened, {"command": "a"},
                             transcripts={"x86_64": "$ true\n"})
        self.assertNotIn("setup_transcript", payload)

    def test_caller_args_dict_is_not_mutated(self) -> None:
        shells, opened = {}, []
        args = {"command": "true", "machine": "arm64"}
        self._call(shells, opened, args)
        self.assertEqual({"command": "true", "machine": "arm64"}, args)

    def test_open_lock_serializes_racing_lazy_opens(self) -> None:
        # parallel role passes share one shells dict
        sessions = self._sessions()
        shells, opened, lock = {}, [], threading.Lock()
        barrier = threading.Barrier(2)

        def open_shell(label):
            opened.append(label)
            time.sleep(0.05)  # widen the check-to-insert window
            return sessions[label], ""

        def call():
            barrier.wait()
            shell_tool.exec_machine_call(
                shells, ("x86_64", "arm64"), open_shell,
                {"command": "true", "machine": "arm64"},
                max_timeout_s=60.0, open_lock=lock,
            )

        threads = [threading.Thread(target=call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(["arm64"], opened)

    def test_explicit_machine_with_single_configured_machine(self) -> None:
        shells, opened = {}, []
        payload = self._call(shells, opened,
                             {"command": "true", "machine": "x86_64"},
                             labels=("x86_64",))
        self.assertEqual("on x86_64\n", payload["stdout"])


class SubstitutionTests(unittest.TestCase):
    def test_number_and_base_ref_placeholders(self) -> None:
        req = {"pull_request": {"number": 23638, "base_ref": "release/7.1"}}
        self.assertEqual(
            "git merge-base release/7.1 fforge/pr/23638",
            pr_review_wrapper.substitute_session_command(
                "git merge-base {base_ref} fforge/pr/{number}", req,
            ),
        )

    def test_missing_metadata_leaves_placeholders(self) -> None:
        self.assertEqual(
            "echo {number}",
            pr_review_wrapper.substitute_session_command("echo {number}", {}),
        )


class PromptSpliceTests(unittest.TestCase):
    def test_shell_roles_user_texts_include_transcript(self) -> None:
        for role in (REVIEWER_ROLE, COMBINER_ROLE,
                     ISSUE_INVESTIGATOR_ROLE, ISSUE_COMBINER_ROLE):
            texts = role.user_texts(_ctx("$ git status --short\nclean\n"))
            joined = "\n".join(texts)
            self.assertIn("already run in your shell container", joined, role.name)
            self.assertIn("$ git status --short", joined, role.name)

    def test_no_transcript_adds_no_block(self) -> None:
        self.assertEqual([], make_session_transcript_texts(_ctx()))
        self.assertEqual(1, len(REVIEWER_ROLE.user_texts(_ctx())))


if __name__ == "__main__":
    unittest.main()
