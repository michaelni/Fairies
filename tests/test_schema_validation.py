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

import openai_pr_review_wrapper as wrapper  # noqa: E402

# Real production output (message trimmed) that slipped past strict mode.
REAL_MISKEYED_REVIEW = {
    "class": "major_request_changes",
    "message": "LLM review of `b5c37a934fe4`: the optional --gcli path ...",
}


class CheckSchemaTests(unittest.TestCase):
    def test_valid_review_passes(self) -> None:
        wrapper.check_schema(
            {"classification": "ok_approve", "message": ""},
            wrapper.REVIEW_SCHEMA["schema"],
        )

    def test_real_miskeyed_review_is_rejected(self) -> None:
        with self.assertRaises(wrapper.SchemaError) as ctx:
            wrapper.check_schema(REAL_MISKEYED_REVIEW, wrapper.REVIEW_SCHEMA["schema"])
        # The error names the offending key rather than crashing opaquely.
        self.assertIn("class", str(ctx.exception))

    def test_bad_enum_value_is_rejected(self) -> None:
        with self.assertRaises(wrapper.SchemaError):
            wrapper.check_schema(
                {"classification": "looks_good", "message": ""},
                wrapper.REVIEW_SCHEMA["schema"],
            )

    def test_wrong_type_is_rejected(self) -> None:
        with self.assertRaises(wrapper.SchemaError):
            wrapper.check_schema(
                {"classification": "ok_approve", "message": 12},
                wrapper.REVIEW_SCHEMA["schema"],
            )

    def test_non_object_is_rejected(self) -> None:
        with self.assertRaises(wrapper.SchemaError):
            wrapper.check_schema("not a dict", wrapper.REVIEW_SCHEMA["schema"])

    def test_generic_over_triage_schema(self) -> None:
        # The same checker validates a dynamically built schema with
        # nullable unions, nested arrays and per-item object schemas.
        schema = wrapper.build_triage_schema(["gpt-5.5"], ["needs-review"])["schema"]
        wrapper.check_schema(
            {
                "route": "engage", "message": "", "reason": "ok",
                "requested_model": None, "requested_effort": "high",
                "label_changes": [
                    {"label": "needs-review", "op": "add",
                     "reason": "x", "post": True},
                ],
            },
            schema,
        )
        with self.assertRaises(wrapper.SchemaError):
            wrapper.check_schema(
                {
                    "route": "engage", "message": "", "reason": "ok",
                    "requested_model": None, "requested_effort": "high",
                    "label_changes": [{"label": "secret", "op": "add",
                                       "reason": "x", "post": True}],
                },
                schema,
            )


class ValidateResultTests(unittest.TestCase):
    def test_real_miskeyed_review_raises_schema_error(self) -> None:
        with self.assertRaises(wrapper.SchemaError):
            wrapper.validate_result(REAL_MISKEYED_REVIEW)

    def test_valid_review_round_trips(self) -> None:
        result = wrapper.validate_result(
            {"classification": "minor_issues_approve", "message": "looks ok"},
        )
        self.assertEqual(result["classification"], "minor_issues_approve")
        self.assertEqual(result["message"], "looks ok")


if __name__ == "__main__":
    unittest.main()
