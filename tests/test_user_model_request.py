"""Tests for user-requested model/effort overrides in triage.

The triage stage emits two optional fields, ``requested_models`` and
``requested_effort``, that propagate a commenter's request to run
specific models (up to two, in parallel) or a reasoning effort in the
main review pass. The schema returned by ``build_triage_schema``
enum-constrains both fields, so ``validate_triage_result`` just passes
them through (deduplicated); the engage branch in ``main()`` then
builds the reviewer lineup from the request via ``make_reviewer``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_pr_review_wrapper as wrapper  # noqa: E402


def _engage(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "route": "engage",
        "message": "",
        "reason": "user explicitly asked for re-review with gpt-5.5",
        "requested_models": [],
        "requested_effort": None,
    }
    base.update(extra)
    return base


class TriageSchemaShapeTests(unittest.TestCase):
    """Pin the schema shape across feature on/off."""

    def test_disabled_schema_omits_override_fields(self) -> None:
        # No --allowed-model -> feature off -> no schema cost.
        schema = wrapper.build_triage_schema([])
        properties = schema["schema"]["properties"]
        required = schema["schema"]["required"]
        self.assertNotIn("requested_models", properties)
        self.assertNotIn("requested_effort", properties)
        self.assertNotIn("requested_models", required)
        self.assertNotIn("requested_effort", required)

    def test_enabled_schema_constrains_models_to_allowlist(self) -> None:
        schema = wrapper.build_triage_schema(["gpt-5.4", "gpt-5.5", "zai:glm-5.2"])
        models_field = schema["schema"]["properties"]["requested_models"]
        # Strict-mode schema is the boundary: the LLM cannot return a
        # model name outside the allowlist, nor more than two entries.
        self.assertEqual(
            models_field["items"]["enum"], ["gpt-5.4", "gpt-5.5", "zai:glm-5.2"],
        )
        self.assertEqual(2, models_field["maxItems"])
        # Required list per strict mode.
        required = schema["schema"]["required"]
        self.assertIn("requested_models", required)
        self.assertIn("requested_effort", required)

    def test_enabled_schema_constrains_effort_to_requestable_set(self) -> None:
        schema = wrapper.build_triage_schema(["gpt-5.5"])
        effort_field = schema["schema"]["properties"]["requested_effort"]
        self.assertEqual(
            effort_field["enum"],
            [None, *wrapper.TRIAGE_REQUESTABLE_EFFORTS],
        )

    def test_requestable_efforts_are_subset_of_cli_choices(self) -> None:
        # The user-requestable effort enum must be a subset of the
        # operator's --reasoning-effort CLI choices, otherwise
        # honoring the override would feed an invalid value into
        # ``responses.create``.
        cli_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
        for effort in wrapper.TRIAGE_REQUESTABLE_EFFORTS:
            self.assertIn(effort, cli_efforts)


class ValidateTriageResultPassthroughTests(unittest.TestCase):
    """The schema is the boundary; validate just extracts the fields."""

    def test_two_models_pass_through_in_request_order(self) -> None:
        result = wrapper.validate_triage_result(
            _engage(requested_models=["zai:glm-5.2", "gpt-5.5"]),
        )
        self.assertEqual(result["requested_models"], ["zai:glm-5.2", "gpt-5.5"])

    def test_duplicate_model_request_is_deduplicated(self) -> None:
        # "gpt-5.5 and gpt-5.5" means one run of gpt-5.5, not two.
        result = wrapper.validate_triage_result(
            _engage(requested_models=["gpt-5.5", "gpt-5.5"]),
        )
        self.assertEqual(result["requested_models"], ["gpt-5.5"])

    def test_effort_passes_through(self) -> None:
        for effort in wrapper.TRIAGE_REQUESTABLE_EFFORTS:
            with self.subTest(effort=effort):
                result = wrapper.validate_triage_result(
                    _engage(requested_effort=effort),
                )
                self.assertEqual(result["requested_effort"], effort)

    def test_missing_fields_get_defaults(self) -> None:
        # When the schema does not include the override fields (no
        # allowlist), the LLM response has no such keys and the
        # validator must default rather than KeyError.
        result = wrapper.validate_triage_result({
            "route": "engage", "message": "", "reason": "ok",
        })
        self.assertEqual(result["requested_models"], [])
        self.assertIsNone(result["requested_effort"])

    def test_non_engage_routes_preserve_override_fields(self) -> None:
        # The override fields are only consumed on the engage path.
        # For helpful_reply / skip we leave the values untouched --
        # they're moot anyway and special-casing them here would just
        # add code without buying anything.
        for route in ("helpful_reply", "skip"):
            with self.subTest(route=route):
                result = wrapper.validate_triage_result({
                    "route": route,
                    "message": "context" if route == "helpful_reply" else "",
                    "reason": "irrelevant",
                    "requested_models": ["gpt-5.5"],
                    "requested_effort": "high",
                })
                self.assertEqual(result["route"], route)
                self.assertEqual(result["requested_models"], ["gpt-5.5"])
                self.assertEqual(result["requested_effort"], "high")


class TriagePromptShapeTests(unittest.TestCase):
    """Pin the prompt-section behavior across allowlist states."""

    def test_no_allowlist_emits_empty_prompt_section(self) -> None:
        # Feature off -> no LLM-visible prompt content for the override.
        self.assertEqual(wrapper.t_prompt_user_request([]), "")

    def test_allowlist_lists_supported_models_and_efforts(self) -> None:
        text = wrapper.t_prompt_user_request(["gpt-5.4", "zai:glm-5.2"])
        self.assertIn("gpt-5.4", text)
        self.assertIn("zai:glm-5.2", text)
        self.assertIn("up to two", text)
        for effort in wrapper.TRIAGE_REQUESTABLE_EFFORTS:
            self.assertIn(effort, text)

    def test_make_triage_developer_prompt_includes_user_request_section(self) -> None:
        prompt = wrapper.make_triage_developer_prompt(
            reviewer_username="bot",
            repo_roots=[],
            vector_store_search_enabled=False,
            web_search_enabled=False,
            code_interpreter_enabled=False,
            podman_shell_enabled=False,
            container_repo_mounts=[],
            allowed_models=["gpt-5.5"],
        )
        self.assertIn("gpt-5.5", prompt)

    def test_make_triage_developer_prompt_off_has_no_override_text(self) -> None:
        # When the feature is off the developer prompt must contain
        # no mention of the override fields so the LLM does not see
        # any conflicting instruction.
        prompt = wrapper.make_triage_developer_prompt(
            reviewer_username="bot",
            repo_roots=[],
            vector_store_search_enabled=False,
            web_search_enabled=False,
            code_interpreter_enabled=False,
            podman_shell_enabled=False,
            container_repo_mounts=[],
        )
        self.assertNotIn("requested_model", prompt)
        self.assertNotIn("requested_effort", prompt)


class PodmanContainerLocationsPromptTests(unittest.TestCase):
    """The podman container has working-tree checkouts under /work, so the
    reviewer must be told where project data lives there (not the OpenAI
    container's bare-repo /mnt/data git-dir paths)."""

    def test_podman_prompt_points_at_work_tree_locations(self) -> None:
        prompt = wrapper.make_developer_prompt(
            source_bundle_attached=False, reviewer_username="bot", repo_roots=[],
            vector_store_search_enabled=False, web_search_enabled=False,
            code_interpreter_enabled=False, podman_shell_enabled=True,
            container_repo_mounts=["/work/ffmpeg", "/work/all_ffmpeg"],
        )
        self.assertIn("/work/all_ffmpeg", prompt)
        self.assertIn("forgejo_git/pulls/", prompt)
        self.assertIn("forgejo_git/issues/", prompt)
        self.assertIn("for_ffmpeg/", prompt)
        self.assertNotIn("/mnt/data/repos", prompt)


if __name__ == "__main__":
    unittest.main()
