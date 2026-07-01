"""Tests for ``common.sanitize_repo_name`` / ``common.dedup_with_suffix``.

These helpers were extracted from ``openai_container.build_container_repo_specs``;
the behaviour they encode (regex-based sanitisation + numeric suffix
collision handling) is what currently determines the in-container
layout that operators see in logs and reports. The tests below pin
that contract so a future edit to either the regex or the suffix
algorithm shows up as a deliberate change rather than a silent drift.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import dedup_with_suffix, sanitize_repo_name  # noqa: E402


class SanitizeRepoNameTests(unittest.TestCase):
    def test_passes_through_safe_chars(self) -> None:
        self.assertEqual("ffmpeg.web-1", sanitize_repo_name("ffmpeg.web-1"))

    def test_replaces_unsafe_chars_with_dash(self) -> None:
        self.assertEqual("messy-name-weird", sanitize_repo_name("messy name!weird"))

    def test_strips_leading_and_trailing_punctuation(self) -> None:
        self.assertEqual("ffmpeg", sanitize_repo_name(".ffmpeg-"))
        self.assertEqual("ffmpeg", sanitize_repo_name("___ffmpeg___"))

    def test_falls_back_to_repo_for_empty_or_all_punct(self) -> None:
        self.assertEqual("repo", sanitize_repo_name(""))
        self.assertEqual("repo", sanitize_repo_name("!!!"))
        self.assertEqual("repo", sanitize_repo_name(".-_"))


class DedupWithSuffixTests(unittest.TestCase):
    def test_unused_returned_as_is_and_recorded(self) -> None:
        used: set[str] = set()
        self.assertEqual("ffmpeg", dedup_with_suffix("ffmpeg", used))
        self.assertIn("ffmpeg", used)

    def test_first_collision_yields_dash_two(self) -> None:
        used: set[str] = {"ffmpeg"}
        self.assertEqual("ffmpeg-2", dedup_with_suffix("ffmpeg", used))

    def test_runs_increment_until_free(self) -> None:
        used: set[str] = {"ffmpeg", "ffmpeg-2", "ffmpeg-3"}
        self.assertEqual("ffmpeg-4", dedup_with_suffix("ffmpeg", used))

    def test_three_repeats_produce_two_three(self) -> None:
        used: set[str] = set()
        names = [dedup_with_suffix("ffmpeg", used) for _ in range(3)]
        self.assertEqual(["ffmpeg", "ffmpeg-2", "ffmpeg-3"], names)


if __name__ == "__main__":
    unittest.main()
