"""Tests for ``fairy.prompt_manual``.

Pins the prompt label shape: the URL must appear verbatim when the
caller knows it, and the bare ``PR #N`` fallback is reserved for the
case where no URL is available.
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


class PromptManualLabelTests(unittest.TestCase):
    def _capture_prompt(self, **kwargs: object) -> str:
        # ``prompt_manual`` renders the prompt to stderr (so the leading
        # blank line stays ordered with the rest of the stderr log) and
        # reads the answer from a bare ``input()``. Feed an empty line
        # ("skip") so it exits on the first iteration, and read back what
        # was written to stderr.
        err = io.StringIO()
        with patch("sys.stderr", err), patch("builtins.input", return_value=""):
            fairy.prompt_manual(**kwargs)
        return err.getvalue()

    def test_url_is_shown_when_provided(self) -> None:
        prompt = self._capture_prompt(
            pr_number=42,
            action="approve",
            pr_url="https://example.test/o/r/pulls/42",
        )
        self.assertIn("https://example.test/o/r/pulls/42", prompt)
        # Bare "PR #42" must NOT also be present -- when the URL is
        # known it is the canonical reference and a "PR #N" duplicate
        # would just clutter the prompt.
        self.assertNotIn("PR #42", prompt)

    def test_falls_back_to_pr_number_when_url_missing(self) -> None:
        prompt = self._capture_prompt(pr_number=42, action="approve")
        self.assertIn("PR #42", prompt)

    def test_falls_back_when_url_is_empty_string(self) -> None:
        prompt = self._capture_prompt(pr_number=42, action="approve", pr_url="")
        self.assertIn("PR #42", prompt)

    def test_action_and_choices_are_present(self) -> None:
        # Keep the rest of the prompt working: the action verb and the
        # choice list are part of the contract with the operator.
        prompt = self._capture_prompt(
            pr_number=42,
            action="request-changes",
            pr_url="https://example.test/o/r/pulls/42",
        )
        self.assertIn("request-changes", prompt)
        self.assertIn("[yes/skip/defer/quit/retry]", prompt)


if __name__ == "__main__":
    unittest.main()
