"""Replay the full wrapper flow with captured request/response pairs.

This test feeds saved stdin payloads into `pr_review_wrapper.main()`
while mocking external services, then verifies the final wrapper JSON output.
It is needed as a top-level regression check so changes in wiring, parsing, or
rendering do not silently break the end-to-end wrapper behavior.

The fixtures here are the wrapper's own debug-response JSON files (the same
format the wrapper writes to ``--debug-response-dir``). To add a new case,
pick a representative file from that directory and copy it into
``tests/fixtures/wrapper_functional_runs/<name>.json`` -- no separate
capture or build tool needed; the debug JSON already contains both
``wrapper_request`` (the stdin object) and ``response`` (the OpenAI
response), which is all this replay needs.
"""
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:  # pragma: no cover - test shim
        pass

    class _FakeError(Exception):
        pass

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.RateLimitError = _FakeError
    fake_openai.AuthenticationError = _FakeError
    fake_openai.InternalServerError = _FakeError
    sys.modules["openai"] = fake_openai

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.dotenv_values = lambda *_args, **_kwargs: {}
    sys.modules["dotenv"] = fake_dotenv

import pr_review_wrapper as wrapper
import openai_reviewer
from llm_review_api import Review


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wrapper_functional_runs"


class _FakeResponses:
    def __init__(self, response_payload: dict[str, object]) -> None:
        self._payload = response_payload
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = dict(kwargs)
        return self._payload


class _FakeOpenAIClient:
    def __init__(
        self,
        response_payload: dict[str, object],
        api_key: str,
        **extra_kwargs: object,
    ) -> None:
        # ``wrapper.main()`` passes ``timeout`` / ``max_retries`` on top of
        # ``api_key``; accept and ignore any other client-construction
        # kwargs so this mock stays compatible when the real constructor
        # signature evolves.
        self.response_payload = response_payload
        self.api_key = api_key
        self.client_kwargs = extra_kwargs
        self.responses = _FakeResponses(response_payload)


class WrapperFunctionalReplayTests(unittest.TestCase):
    def test_replays_fixture_through_main(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("*.json"))
        self.assertTrue(fixture_paths, "no wrapper functional fixtures found")

        for fixture_path in fixture_paths:
            with self.subTest(fixture=fixture_path.name):
                fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                request_obj = fixture["wrapper_request"]
                response_obj = fixture["response"]

                stdin_payload = json.dumps(request_obj, ensure_ascii=False)
                stdout_buffer = io.StringIO()

                fake_client_holder: dict[str, _FakeOpenAIClient] = {}

                def fake_openai_ctor(*, api_key: str, **extra_kwargs: object) -> _FakeOpenAIClient:
                    client = _FakeOpenAIClient(response_obj, api_key=api_key, **extra_kwargs)
                    fake_client_holder["client"] = client
                    return client

                with (
                    mock.patch.object(wrapper, "OpenAI", side_effect=fake_openai_ctor),
                    mock.patch.object(wrapper, "load_api_key", return_value="test-key"),
                    mock.patch.object(wrapper, "upload_text_file", side_effect=["file-patch", "file-source"]),
                    mock.patch.object(wrapper, "delete_uploaded_file", return_value=None),
                    mock.patch.object(wrapper, "find_repo_root", return_value=Path.cwd()),
                    mock.patch.object(wrapper, "get_all_repo_roots", return_value=[Path.cwd()]),
                    mock.patch.object(wrapper.sys, "argv", ["pr_review_wrapper.py", "--model", "openai:gpt-5.4", "--no-source-bundle"]),
                    mock.patch.object(wrapper.sys, "stdin", io.StringIO(stdin_payload)),
                    mock.patch.object(wrapper.sys, "stdout", stdout_buffer),
                ):
                    exit_code = wrapper.main()

                self.assertEqual(0, exit_code)
                self.assertIn("client", fake_client_holder)

                output_obj = json.loads(stdout_buffer.getvalue())
                self.assertIn("classification", output_obj)
                self.assertIn("message", output_obj)
                self.assertIsInstance(output_obj["message"], str)
                self.assertNotIn("\uE200", output_obj["message"])
                self.assertNotIn("\uE201", output_obj["message"])
                self.assertIn("Sources:", output_obj["message"])


class _MainPassReached(Exception):
    """Sentinel: control reached the main reviewer responses.create call."""


def _run_wrapper_with_stubbed_triage(
    request_obj: dict,
    triage_result: dict,
    review_pr_stub: object = None,
) -> tuple[int | None, str, bool]:
    """Run ``wrapper.main()`` with external services mocked and the triage
    stage stubbed to ``triage_result``.

    Returns ``(exit_code, stdout, main_pass_reached)``. ``main_pass_reached``
    is True iff control reached the main reviewer ``responses.create`` call
    (a sentinel is raised there), in which case ``exit_code`` is None.
    ``review_pr_stub``, when given, replaces ``wrapper.review_pr`` so the
    main pass completes with a canned ``Review`` instead of hitting the
    sentinel.
    """
    fixture = json.loads(
        (FIXTURE_DIR / "sample_multi_source_filecite.json").read_text(encoding="utf-8")
    )

    def fake_ctor(*, api_key: str, **extra: object) -> _FakeOpenAIClient:
        return _FakeOpenAIClient(fixture["response"], api_key=api_key, **extra)

    stdout_buffer = io.StringIO()
    main_pass_reached = False

    def sentinel_create(*_a: object, **_kw: object) -> object:
        nonlocal main_pass_reached
        main_pass_reached = True
        raise _MainPassReached()

    with (
        mock.patch.object(wrapper, "OpenAI", side_effect=fake_ctor),
        mock.patch.object(wrapper, "load_api_key", return_value="test-key"),
        mock.patch.object(wrapper, "upload_text_file", side_effect=["file-patch", "file-source"]),
        mock.patch.object(wrapper, "delete_uploaded_file", return_value=None),
        mock.patch.object(wrapper, "find_repo_root", return_value=Path.cwd()),
        mock.patch.object(wrapper, "get_all_repo_roots", return_value=[Path.cwd()]),
        mock.patch.object(wrapper, "run_triage", return_value=triage_result),
        mock.patch.object(wrapper, "review_pr", side_effect=review_pr_stub or wrapper.review_pr),
        # The main pass runs inside OpenAIReviewer, so the sentinel must
        # intercept openai_reviewer's namespace, not the wrapper's.
        mock.patch.object(openai_reviewer, "call_with_rate_limit_retry", side_effect=sentinel_create),
        mock.patch.object(
            wrapper.sys, "argv",
            ["pr_review_wrapper.py", "--model", "openai:gpt-5.4", "--no-source-bundle", "--triage-model", "openai:gpt-x"],
        ),
        mock.patch.object(wrapper.sys, "stdin", io.StringIO(json.dumps(request_obj))),
        mock.patch.object(wrapper.sys, "stdout", stdout_buffer),
    ):
        try:
            exit_code: int | None = wrapper.main()
        except _MainPassReached:
            exit_code = None
    return exit_code, stdout_buffer.getvalue(), main_pass_reached


def _fixture_request(**extra: object) -> dict:
    fixture = json.loads(
        (FIXTURE_DIR / "sample_multi_source_filecite.json").read_text(encoding="utf-8")
    )
    request_obj = dict(fixture["wrapper_request"])
    request_obj.update(extra)
    return request_obj


class TriageSkipOverrideTests(unittest.TestCase):
    """``ignore_triage_skip`` makes the wrapper run the main pass on skip.

    The flag is fairy's --force-review-skip crossing the
    process boundary. Triage is stubbed to vote ``skip``; the only
    difference between the two runs is the request flag, so this pins
    that the wrapper honors it rather than always short-circuiting on a
    skip verdict.
    """

    TRIAGE_SKIP = {
        "route": "skip",
        "reason": "already approved on this head",
        "label_changes": [],
    }

    def test_skip_honored_without_flag(self) -> None:
        exit_code, out, main_pass_reached = _run_wrapper_with_stubbed_triage(
            _fixture_request(), self.TRIAGE_SKIP,
        )
        self.assertFalse(main_pass_reached)
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(out)["classification"], "skip")

    def test_skip_overridden_with_flag(self) -> None:
        _exit_code, out, main_pass_reached = _run_wrapper_with_stubbed_triage(
            _fixture_request(ignore_triage_skip=True), self.TRIAGE_SKIP,
        )
        self.assertTrue(main_pass_reached)
        self.assertEqual(out, "")


class CiTriageEngageTests(unittest.TestCase):
    """Routing on a CI-red request (``ci_triage`` present).

    ``engage`` reaches the main reviewer pass even on red CI (the old code
    forced such requests to skip). ``reply_no_verdict`` still short-circuits,
    and ``force_engage`` (fairy's --force-engage) makes the
    wrapper run the full pass regardless of the triage route.
    """

    CI_TRIAGE = {"head_ref": "pr-head", "failures": ["Test / Fate"]}
    TRIAGE_ENGAGE = {"route": "engage", "reason": "worth a look", "label_changes": []}
    TRIAGE_REPLY = {
        "route": "reply_no_verdict",
        "reason": "tree is red",
        "message": "Heads-up: CI is red.",
        "label_changes": [],
    }

    def test_engage_on_red_ci_reaches_main_pass(self) -> None:
        _exit, out, reached = _run_wrapper_with_stubbed_triage(
            _fixture_request(ci_triage=self.CI_TRIAGE), self.TRIAGE_ENGAGE,
        )
        self.assertTrue(reached)
        self.assertEqual(out, "")

    def test_reply_no_verdict_short_circuits_on_red_ci(self) -> None:
        exit_code, out, reached = _run_wrapper_with_stubbed_triage(
            _fixture_request(ci_triage=self.CI_TRIAGE), self.TRIAGE_REPLY,
        )
        self.assertFalse(reached)
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(out)["classification"], "reply_no_verdict")

    def test_force_engage_overrides_reply_no_verdict_on_red_ci(self) -> None:
        _exit, out, reached = _run_wrapper_with_stubbed_triage(
            _fixture_request(ci_triage=self.CI_TRIAGE, force_engage=True),
            self.TRIAGE_REPLY,
        )
        self.assertTrue(reached)
        self.assertEqual(out, "")


class EngageLabelOwnershipTests(unittest.TestCase):
    """On engage the final verdict author owns the labels: the wrapper
    emits the review's label_changes and discards the triage guesses."""

    TRIAGE_ENGAGE_WITH_LABELS = {
        "route": "engage",
        "reason": "worth a look",
        "label_changes": [
            {"label": "enhancement", "op": "add", "reason": "triage guess", "post": False},
        ],
    }

    def test_reviewer_labels_emitted_triage_labels_discarded(self) -> None:
        reviewer_labels = (
            {"label": "needs docs", "op": "add", "reason": "doc mismatch", "post": True},
        )
        seen: dict[str, object] = {}

        def stub(ctx: object, reviewers: list, combiner: object) -> Review:
            seen["reviewers"] = reviewers
            seen["combiner"] = combiner
            return Review(
                "minor_issues_approve", "docs drifted",
                label_changes=reviewer_labels, model="openai:x",
            )

        exit_code, out, _reached = _run_wrapper_with_stubbed_triage(
            _fixture_request(triage_label_allowlist=["needs docs", "enhancement"]),
            self.TRIAGE_ENGAGE_WITH_LABELS,
            review_pr_stub=stub,
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(out)["label_changes"], list(reviewer_labels))
        # Single-reviewer run: the reviewer role itself owns label_changes.
        (reviewer,) = seen["reviewers"]
        self.assertIn("label_changes", reviewer.role.schema["schema"]["properties"])
        self.assertIsNone(seen["combiner"])


if __name__ == "__main__":
    unittest.main()
