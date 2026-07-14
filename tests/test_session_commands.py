"""--session-command: pre-run shell commands spliced into the prompt."""
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import podman_host
import pr_review_wrapper
import shell_tool
from llm_prompt import REVIEWER_ROLE, make_session_transcript_texts
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
    def test_reviewer_user_texts_include_transcript(self) -> None:
        texts = REVIEWER_ROLE.user_texts(_ctx("$ git status --short\nclean\n"))
        joined = "\n".join(texts)
        self.assertIn("already run in your shell container", joined)
        self.assertIn("$ git status --short", joined)

    def test_no_transcript_adds_no_block(self) -> None:
        self.assertEqual([], make_session_transcript_texts(_ctx()))
        self.assertEqual(1, len(REVIEWER_ROLE.user_texts(_ctx())))


if __name__ == "__main__":
    unittest.main()
