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

LLM skip decisions carry the model's own explanation as the reason.

Regression: the TUI/manual detail showed "reason LLM chose skip" -- a
narration of what skip means -- while the triager's actual reason was
logged and thrown away.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import issue_fairy  # noqa: E402

# Shaped like a real triage skip reason (they arrive as the review message
# since pr_review_wrapper forwards triage_result["reason"] on route=skip).
REASON = ("PR only rewraps a code comment; no functional change for the "
          "reviewer to assess.")


class SkipReasonTests(unittest.TestCase):
    def test_pr_skip_reason_is_the_models_explanation(self) -> None:
        d = fairy.decision_from_review(
            fairy.LLMReview("skip", REASON),
            number=1, title="t", author="a", auto_merge="-",
            last_activity=None, base_reason="review",
        )
        self.assertEqual(d.reason, f"LLM skip: {REASON}")
        self.assertEqual(d.llm_message, REASON)

    def test_issue_skip_reason_is_the_models_explanation(self) -> None:
        prepared = issue_fairy.PreparedIssue(
            issue={}, number=2, title="t", author="a", last_activity=None,
            base_reason="stale", discussion=[], reviewer_username=None,
        )
        d = issue_fairy.issue_decision_from_review(
            prepared, fairy.LLMReview("skip", REASON))
        self.assertEqual(d.reason, f"LLM skip: {REASON}")

    def test_fallbacks(self) -> None:
        self.assertEqual(fairy.llm_skip_reason(""), "LLM chose skip")
        self.assertEqual(fairy.llm_skip_reason("first line\nsecond"),
                         "LLM skip: first line")
        self.assertEqual(len(fairy.llm_skip_reason("x" * 500)), 400)


if __name__ == "__main__":
    unittest.main()
