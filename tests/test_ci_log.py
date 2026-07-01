"""Tests for ci_log (CI job-log tail fetch) and its wiring in fairy.

A Forgejo commit-status ``target_url`` for an Actions job points at the web
UI; the plain-text log lives at that page's ``attempt/<n>/logs`` sub-route
and is served by an unauthenticated GET (no token, no anti-bot challenge,
``Accept-Ranges: bytes``). ``ci_log`` fetches the tail of that log and
``fairy.attach_ci_failure_logs`` folds it into each failing
context as ``log_tail`` -- but only once a triager/reviewer is known to run.

These tests pin the logic worth regressing without hitting the network: the
job-link recogniser, the tail extractor (including the Range-sliced
partial-first-line drop), the no-network guard rails, and the enrichment
mutating the exact dicts ``build_ci_triage_payload`` re-exports.

Real-data anchor: ``fixtures/ci_log/run59352_job0_excerpt.txt`` is a verbatim
excerpt of the failing log of PR 23510 / run 59352 / job 0 (captured
2026-06-21), whose real error is ``gcc: fatal error: cannot execute
'cc1obj'`` -- exactly the kind of detail the bot otherwise never sees.
"""

from __future__ import annotations

import argparse
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ci_log  # noqa: E402
import fairy  # noqa: E402


REAL_TARGET_URL = "https://code.ffmpeg.org/FFmpeg/FFmpeg/actions/runs/59352/jobs/0"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ci_log" / "run59352_job0_excerpt.txt"


class JobUrlRecogniserTests(unittest.TestCase):
    def test_matches_real_job_target_url(self) -> None:
        self.assertIsNotNone(ci_log._JOB_URL_RE.match(REAL_TARGET_URL))

    def test_matches_attempt_logs_url(self) -> None:
        self.assertIsNotNone(
            ci_log._JOB_URL_RE.match(REAL_TARGET_URL + "/attempt/1/logs")
        )

    def test_rejects_path_only_url(self) -> None:
        # Needs an absolute http(s) URL; a bare path has no host to GET.
        self.assertIsNone(
            ci_log._JOB_URL_RE.match("/FFmpeg/FFmpeg/actions/runs/59352/jobs/0")
        )

    def test_rejects_run_without_job(self) -> None:
        self.assertIsNone(
            ci_log._JOB_URL_RE.match(
                "https://code.ffmpeg.org/FFmpeg/FFmpeg/actions/runs/59352"
            )
        )

    def test_rejects_unrelated_url(self) -> None:
        self.assertIsNone(ci_log._JOB_URL_RE.match("https://ci.example.org/build/123"))


class ExtractLogTailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = FIXTURE.read_bytes()
        self.lines = self.body.decode().splitlines()

    def test_returns_last_n_lines(self) -> None:
        tail = ci_log._extract_log_tail(self.body, partial=False, max_lines=5)
        self.assertEqual(tail, "\n".join(self.lines[-5:]))

    def test_full_excerpt_when_n_exceeds_line_count(self) -> None:
        tail = ci_log._extract_log_tail(self.body, partial=False, max_lines=10_000)
        self.assertEqual(tail, "\n".join(self.lines))

    def test_real_error_present_in_tail(self) -> None:
        tail = ci_log._extract_log_tail(self.body, partial=False, max_lines=10_000)
        assert tail is not None
        self.assertIn("cannot execute 'cc1obj'", tail)

    def test_partial_discards_fragment_first_line(self) -> None:
        # A Range slice opens mid-line; that fragment must not survive.
        sliced = b"ute 'cc1obj': error\nmake: *** Error 1\n"
        tail = ci_log._extract_log_tail(sliced, partial=True, max_lines=10)
        self.assertEqual(tail, "make: *** Error 1")

    def test_kept_first_line_when_not_partial(self) -> None:
        tail = ci_log._extract_log_tail(b"first\nsecond\n", partial=False, max_lines=10)
        self.assertEqual(tail, "first\nsecond")

    def test_empty_body_is_none(self) -> None:
        self.assertIsNone(ci_log._extract_log_tail(b"", partial=False, max_lines=10))


class FetchGuardrailTests(unittest.TestCase):
    """A non-job URL must short-circuit before any network access."""

    def test_non_job_url_returns_none(self) -> None:
        self.assertIsNone(
            ci_log.fetch_job_log_tail("https://ci.example.org/build/1", max_lines=50)
        )


class FetchErrorStringTests(unittest.TestCase):
    """A failed fetch surfaces an ``error fetching`` notice, not None, so the
    LLM sees the attempt failed."""

    def test_http_error_returns_error_string(self) -> None:
        err = urllib.error.HTTPError(
            REAL_TARGET_URL, 404, "Not Found", {}, io.BytesIO(b"logs have been cleaned up")
        )
        with mock.patch.object(ci_log.urllib.request, "urlopen", side_effect=err):
            out = ci_log.fetch_job_log_tail(REAL_TARGET_URL, max_lines=50)
        self.assertEqual(
            out, f'error fetching "{REAL_TARGET_URL}": HTTP 404 (logs have been cleaned up)'
        )

    def test_url_error_returns_error_string(self) -> None:
        with mock.patch.object(
            ci_log.urllib.request, "urlopen", side_effect=urllib.error.URLError("boom"),
        ):
            out = ci_log.fetch_job_log_tail(REAL_TARGET_URL, max_lines=50)
        assert out is not None
        self.assertTrue(out.startswith(f'error fetching "{REAL_TARGET_URL}": '))
        self.assertIn("boom", out)


class AttachCiFailureLogsTests(unittest.TestCase):
    def _details(self) -> list[dict[str, object]]:
        return [
            {"context": "Test / Fate (linux-amd64, static, 32 bit) (pull_request)",
             "state": "FAILURE", "target_url": REAL_TARGET_URL,
             "description": "Failing after 26s"},
            {"context": "no-link", "state": "FAILURE", "target_url": "", "description": "x"},
        ]

    def test_attaches_tail_to_linked_context_only(self) -> None:
        args = argparse.Namespace(ci_failure_log_lines=300)
        with mock.patch.object(
            fairy.ci_log, "fetch_job_log_tail", return_value="boom\nError 1",
        ) as fetch:
            details = self._details()
            fairy.attach_ci_failure_logs(args, details)
        fetch.assert_called_once_with(REAL_TARGET_URL, max_lines=300)
        self.assertEqual(details[0]["log_tail"], "boom\nError 1")
        self.assertNotIn("log_tail", details[1])

    def test_disabled_when_zero_lines(self) -> None:
        args = argparse.Namespace(ci_failure_log_lines=0)
        with mock.patch.object(fairy.ci_log, "fetch_job_log_tail") as fetch:
            details = self._details()
            fairy.attach_ci_failure_logs(args, details)
        fetch.assert_not_called()
        self.assertNotIn("log_tail", details[0])

    def test_unreachable_log_leaves_context_unchanged(self) -> None:
        args = argparse.Namespace(ci_failure_log_lines=300)
        with mock.patch.object(
            fairy.ci_log, "fetch_job_log_tail", return_value=None,
        ):
            details = self._details()
            fairy.attach_ci_failure_logs(args, details)
        self.assertNotIn("log_tail", details[0])

    def test_payload_reexports_attached_tail(self) -> None:
        # build_ci_triage_payload stores the same dicts; enriching them after
        # the payload is built must still surface in failure_contexts.
        args = argparse.Namespace(ci_failure_log_lines=300)
        details = self._details()
        payload = fairy.build_ci_triage_payload("deadbeef", details, [])
        with mock.patch.object(
            fairy.ci_log, "fetch_job_log_tail", return_value="tail-text",
        ):
            fairy.attach_ci_failure_logs(args, details)
        contexts = payload["failure_contexts"]
        assert isinstance(contexts, list)
        self.assertEqual(contexts[0]["log_tail"], "tail-text")


if __name__ == "__main__":
    unittest.main()
