"""Tests for triage label change wiring (the per-label ``label_changes`` form)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402
import llm_review_api  # noqa: E402
import pr_review_wrapper as wrapper  # noqa: E402
from llm_prompt import t_prompt_triage_labels  # noqa: E402
import fairy as paa  # noqa: E402


def _triage_result(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "route": "skip",
        "message": "",
        "reason": "nothing to do",
        "label_changes": [],
    }
    base.update(extra)
    return base


def _change(label: str, op: str, reason: str = "", post: bool = False) -> dict[str, object]:
    return {"label": label, "op": op, "reason": reason, "post": post}


class TriageLabelSchemaTests(unittest.TestCase):
    def test_disabled_schema_omits_label_field(self) -> None:
        schema = llm_review_api.build_triage_schema([], [])
        self.assertNotIn("label_changes", schema["schema"]["properties"])
        self.assertNotIn("label_changes", schema["schema"]["required"])

    def test_enabled_schema_constrains_labels_to_allowlist(self) -> None:
        schema = llm_review_api.build_triage_schema([], ["needs-review", "stale"])
        item = schema["schema"]["properties"]["label_changes"]["items"]
        self.assertEqual(item["properties"]["label"]["enum"], ["needs-review", "stale"])
        self.assertEqual(item["properties"]["op"]["enum"], ["add", "remove"])
        self.assertIn("label_changes", schema["schema"]["required"])

    def test_each_change_requires_reason_and_post(self) -> None:
        # Forcing reason + post per change is what makes a label both
        # auditable (reason) and self-explanatory on the PR when needed
        # (post); both must be required alongside label/op.
        schema = llm_review_api.build_triage_schema([], ["needs-review"])
        required = schema["schema"]["properties"]["label_changes"]["items"]["required"]
        self.assertEqual(set(required), {"label", "op", "reason", "post"})


class SanitizeLabelChangesTests(unittest.TestCase):
    def test_valid_change_passes_through(self) -> None:
        out = llm_review_api.sanitize_label_changes(
            [_change("needs-review", "add", "a maintainer asked", post=True)],
            ["needs-review", "stale"],
        )
        self.assertEqual(out, [
            {"label": "needs-review", "op": "add", "reason": "a maintainer asked", "post": True},
        ])

    def test_unknown_label_dropped(self) -> None:
        out = llm_review_api.sanitize_label_changes(
            [_change("secret", "add"), _change("stale", "remove")],
            ["needs-review", "stale"],
        )
        self.assertEqual([c["label"] for c in out], ["stale"])

    def test_bad_op_dropped(self) -> None:
        out = llm_review_api.sanitize_label_changes(
            [_change("stale", "toggle")], ["stale"],
        )
        self.assertEqual(out, [])

    def test_no_allowlist_yields_empty(self) -> None:
        out = llm_review_api.sanitize_label_changes([_change("needs-review", "add")], [])
        self.assertEqual(out, [])

    def test_duplicate_label_op_deduped(self) -> None:
        out = llm_review_api.sanitize_label_changes(
            [_change("stale", "add"), _change("stale", "add")], ["stale"],
        )
        self.assertEqual(len(out), 1)

    def test_missing_reason_and_post_default(self) -> None:
        out = llm_review_api.sanitize_label_changes([{"label": "stale", "op": "add"}], ["stale"])
        self.assertEqual(out, [{"label": "stale", "op": "add", "reason": "", "post": False}])


class ValidateTriageLabelTests(unittest.TestCase):
    def test_label_changes_pass_through(self) -> None:
        result = llm_review_api.validate_triage_result(
            _triage_result(label_changes=[_change("needs-review", "add", "x", post=True)]),
            allowed_labels=["needs-review", "stale"],
        )
        self.assertEqual(result["label_changes"], [
            {"label": "needs-review", "op": "add", "reason": "x", "post": True},
        ])

    def test_unknown_label_changes_dropped(self) -> None:
        result = llm_review_api.validate_triage_result(
            _triage_result(label_changes=[_change("secret", "add"), _change("stale", "remove")]),
            allowed_labels=["needs-review", "stale"],
        )
        self.assertEqual([c["label"] for c in result["label_changes"]], ["stale"])

    def test_no_allowlist_yields_empty(self) -> None:
        result = llm_review_api.validate_triage_result(
            _triage_result(label_changes=[_change("needs-review", "add")]),
        )
        self.assertEqual(result["label_changes"], [])


class EmitReviewStdoutTests(unittest.TestCase):
    def _emit(self, **kwargs: object) -> dict[str, object]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wrapper.emit_review_stdout("skip", "", **kwargs)
        return json.loads(buf.getvalue())

    def test_label_changes_emitted(self) -> None:
        out = self._emit(label_changes=[_change("needs sample", "add", "x", post=True)])
        self.assertEqual(out["label_changes"][0]["label"], "needs sample")

    def test_no_label_changes_key_when_empty(self) -> None:
        out = self._emit()
        self.assertNotIn("label_changes", out)


class TriageLabelPromptTests(unittest.TestCase):
    def test_no_allowlist_emits_empty_section(self) -> None:
        self.assertEqual(t_prompt_triage_labels([]), "")

    def test_allowlist_lists_labels(self) -> None:
        text = t_prompt_triage_labels(["needs-review", "stale"])
        self.assertIn("needs-review", text)
        self.assertIn("stale", text)
        self.assertIn("label_changes", text)
        self.assertIn("post", text)

    def test_definitions_filtered_to_allowlist(self) -> None:
        # Regression (#23293): the model was shown the "needs testing"
        # definition even though only "needs docs" was allowed.
        text = t_prompt_triage_labels(["fix/bug", "needs docs"])
        self.assertIn("Label: needs docs,", text)
        self.assertNotIn("Label: needs testing,", text)
        self.assertNotIn("Label: needs sample,", text)
        self.assertNotIn("Label: enhancement,", text)


class TriageLabelAllowlistFromRequestTests(unittest.TestCase):
    def test_payload_allowlist_extracted(self) -> None:
        request = {"triage_label_allowlist": ["a", "b", 3, ""]}
        self.assertEqual(
            wrapper.triage_label_allowlist_from_request(request),
            ["a", "b"],
        )


class PrAutoApproveLabelParsingTests(unittest.TestCase):
    def test_parse_label_changes_filters_allowlist_and_op(self) -> None:
        parsed = paa.parse_label_changes(
            [
                _change("needs-review", "add", "r"),
                _change("other", "add"),
                _change("needs-review", "toggle"),
            ],
            ["needs-review"],
        )
        self.assertEqual(parsed, (paa.LabelChange("needs-review", "add", "r", False),))

    def test_manual_action_description_includes_labels(self) -> None:
        decision = paa.Decision(
            42, "title", "author", "-", "skip", "LLM chose skip", None,
            "skip", "",
            label_changes=(paa.LabelChange("needs-review", "add"),),
        )
        self.assertIn("needs-review", paa.manual_action_description(decision))

    def test_manual_action_description_includes_label_justification(self) -> None:
        # The triage justification must reach the operator-facing apply line
        # so a label like "needs sample" can be judged in context.
        decision = paa.Decision(
            42, "title", "author", "-", "skip", "LLM chose skip", None,
            "skip", "",
            label_changes=(paa.LabelChange(
                "needs sample", "add",
                "repro requires a clip the author omitted",
            ),),
        )
        desc = paa.manual_action_description(decision)
        self.assertIn("needs sample", desc)
        self.assertIn("repro requires a clip the author omitted", desc)


class PostLabelExplanationsTests(unittest.TestCase):
    """The rationale is posted only for labels that actually transition this
    run, so the PR's label set is the natural idempotency key and no separate
    dedup state is required (a re-requested add of an already-present label
    posts nothing)."""

    def _decision(self, *changes: paa.LabelChange) -> paa.Decision:
        return paa.Decision(
            7, "t", "a", "-", "skip", "r", None, "skip", "",
            label_changes=tuple(changes),
        )

    def _run(self, decision: paa.Decision, current: set[str]) -> list[str]:
        args = argparse.Namespace(owner="o", repo="r")
        with mock.patch.object(paa, "post_issue_comment") as post:
            paa.post_label_explanations(args, decision, current)
        return [call.args[4] for call in post.call_args_list]

    def test_posts_on_real_add_transition(self) -> None:
        bodies = self._run(
            self._decision(paa.LabelChange("needs sample", "add", "need a clip", post=True)),
            current=set(),
        )
        self.assertEqual(len(bodies), 1)
        self.assertIn("needs sample", bodies[0])
        self.assertIn("need a clip", bodies[0])

    def test_no_post_when_label_already_present(self) -> None:
        # The label is already on the PR -> no transition -> no comment,
        # even though the triager re-requested the add with post=True.
        bodies = self._run(
            self._decision(paa.LabelChange("needs sample", "add", "need a clip", post=True)),
            current={"needs sample"},
        )
        self.assertEqual(bodies, [])

    def test_no_post_when_post_flag_false(self) -> None:
        bodies = self._run(
            self._decision(paa.LabelChange("stale", "add", "log only", post=False)),
            current=set(),
        )
        self.assertEqual(bodies, [])

    def test_posts_on_real_remove_transition_only(self) -> None:
        present = self._run(
            self._decision(paa.LabelChange("stale", "remove", "author replied", post=True)),
            current={"stale"},
        )
        self.assertEqual(len(present), 1)
        absent = self._run(
            self._decision(paa.LabelChange("stale", "remove", "author replied", post=True)),
            current=set(),
        )
        self.assertEqual(absent, [])


class ApplyIssueLabelChangesTests(unittest.TestCase):
    def test_uses_gcli_pulls_labels(self) -> None:
        args = argparse.Namespace(gcli_account=None, forge_type="gitea", verbose=0)
        with mock.patch("forge_gcli.run_cmd") as run_cmd:
            run_cmd.return_value = subprocess.CompletedProcess([], 0, "", "")
            forge_gcli.apply_issue_label_changes(
                args, "owner", "repo", 42, ["needs-review"], ["stale"], {"stale"},
            )
        self.assertEqual(
            run_cmd.call_args[0][0],
            [
                "gcli", "-t", "gitea",
                "pulls", "-o", "owner", "-r", "repo", "-i", "42", "labels",
                "add", "needs-review",
                "remove", "stale",
            ],
        )

    def test_skips_remove_when_label_absent(self) -> None:
        args = argparse.Namespace(gcli_account=None, forge_type="gitea", verbose=0)
        with mock.patch("forge_gcli.run_cmd") as run_cmd:
            run_cmd.return_value = subprocess.CompletedProcess([], 0, "", "")
            forge_gcli.apply_issue_label_changes(
                args, "owner", "repo", 42, [], ["stale"], set(),
            )
        run_cmd.assert_not_called()


if __name__ == "__main__":
    unittest.main()
