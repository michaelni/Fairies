"""Generic schema validation of model output.

OpenAI ``strict`` structured output is best-effort, not guaranteed, so
``check_schema`` re-validates every response against the exact schema we
sent. The malformed reviewer fixture below is the real output that broke
production (PR #23410, resp_07a40beaaa31ca6b...): the model emitted the
key ``class`` instead of ``classification``, which the old hand-rolled
checks turned into ``RuntimeError: invalid classification: None`` and an
unhandled traceback.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_review_api  # noqa: E402

# Real production output (message trimmed) that slipped past strict mode.
REAL_MISKEYED_REVIEW = {
    "class": "major_request_changes",
    "message": "LLM review of `b5c37a934fe4`: the optional --gcli path ...",
}


class CheckSchemaTests(unittest.TestCase):
    def test_valid_review_passes(self) -> None:
        llm_review_api.check_schema(
            {"classification": "ok_approve", "message": "",
             "head_vs_branch_diff_evidence": False},
            llm_review_api.REVIEW_SCHEMA["schema"],
        )

    def test_real_miskeyed_review_is_rejected(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError) as ctx:
            llm_review_api.check_schema(REAL_MISKEYED_REVIEW, llm_review_api.REVIEW_SCHEMA["schema"])
        # The error names the offending key rather than crashing opaquely.
        self.assertIn("class", str(ctx.exception))

    def test_bad_enum_value_is_rejected(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {"classification": "looks_good", "message": "",
                 "head_vs_branch_diff_evidence": False},
                llm_review_api.REVIEW_SCHEMA["schema"],
            )

    def test_wrong_type_is_rejected(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {"classification": "ok_approve", "message": 12,
                 "head_vs_branch_diff_evidence": False},
                llm_review_api.REVIEW_SCHEMA["schema"],
            )

    def test_non_object_is_rejected(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema("not a dict", llm_review_api.REVIEW_SCHEMA["schema"])

    def test_generic_over_triage_schema(self) -> None:
        # The same checker validates a dynamically built schema with
        # nullable unions, nested arrays and per-item object schemas.
        schema = llm_review_api.build_triage_schema(["gpt-5.5"], ["needs-review"])["schema"]
        llm_review_api.check_schema(
            {
                "route": "engage", "message": "", "reason": "ok",
                "prompt_injection": False,
                "requested_models": [], "requested_effort": "high",
                "label_changes": [
                    {"label": "needs-review", "op": "add",
                     "reason": "x", "post": True},
                ],
            },
            schema,
        )
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {
                    "route": "engage", "message": "", "reason": "ok",
                    "prompt_injection": False,
                    "requested_models": [], "requested_effort": "high",
                    "label_changes": [{"label": "secret", "op": "add",
                                       "reason": "x", "post": True}],
                },
                schema,
            )

    def test_max_items_is_enforced(self) -> None:
        # requested_models advertises maxItems=2 to the API; the local
        # checker must enforce the same bound to stay in lockstep.
        schema = llm_review_api.build_triage_schema(["gpt-5.4", "gpt-5.5", "zai:glm-5.2"])["schema"]
        base = {"route": "engage", "message": "", "reason": "ok",
                "prompt_injection": False, "requested_effort": None}
        llm_review_api.check_schema(
            {**base, "requested_models": ["gpt-5.4", "zai:glm-5.2"]}, schema,
        )
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {**base, "requested_models": ["gpt-5.4", "gpt-5.5", "zai:glm-5.2"]},
                schema,
            )


class ValidateReviewTests(unittest.TestCase):
    def test_real_miskeyed_review_raises_schema_error(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.validate_review(REAL_MISKEYED_REVIEW)

    def test_valid_review_round_trips(self) -> None:
        result = llm_review_api.validate_review(
            {"classification": "minor_issues_approve", "message": "looks ok",
             "head_vs_branch_diff_evidence": False},
        )
        self.assertEqual(result["classification"], "minor_issues_approve")
        self.assertEqual(result["message"], "looks ok")

    def test_self_reported_diff_evidence_raises(self) -> None:
        # Real bogus verdict (PR #23553, resp_0b30b334ceafa773...): the
        # blocking issue rested on diffing the PR head against the master
        # tip. With the flag set, the verdict must be rejected so the
        # draft is dropped / the pass retried.
        with self.assertRaises(llm_review_api.SelfReportedViolation) as ctx:
            llm_review_api.validate_review({
                "classification": "major_request_changes",
                "message": "LLM-GPT-5.4: this patch drops the current "
                           "master fix for reference-only resource reuse.",
                "head_vs_branch_diff_evidence": True,
            })
        self.assertIn("major_request_changes", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
