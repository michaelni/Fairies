"""LLM skip decisions carry the model's own explanation as the reason.

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
