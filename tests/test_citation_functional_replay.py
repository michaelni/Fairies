"""Replay real OpenAI responses through citation rendering.

This test loads captured response fixtures, runs the user-visible ``message``
field of each through ``render_file_citations_for_markdown``, and checks that
no OpenAI control markers leak into the rendered output and that responses
carrying ``file_citation`` annotations end with a ``Sources:`` footer. It is
needed to catch regressions that only show up with real response payload
shapes.

The renderer (not ``validate_result``) is exercised directly so that both
review and triage envelopes are covered. Both schemas embed the user-facing
text under a top-level ``message`` field; the schema-specific envelope
validation is covered by other tests.
"""
import json
import unittest
from pathlib import Path

import openai_reviewer
from openai_common import extract_response_text


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "openai_citation_runs"


class CitationFunctionalReplayTests(unittest.TestCase):
    def test_replay_real_openai_payloads(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("*.json"))
        if not fixture_paths:
            self.skipTest(
                "No functional fixtures yet. Collect with "
                "`python tools/openai_collect_citation_fixtures.py "
                "--input-dir <openai_debug_dir>`",
            )

        rendered_count = 0
        skipped_no_message = 0
        for fixture_path in fixture_paths:
            with self.subTest(fixture=fixture_path.name):
                fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                response = fixture.get("response")
                self.assertIsInstance(response, dict, "fixture missing response object")

                raw_text = extract_response_text(response)
                try:
                    parsed = json.loads(raw_text)
                except json.JSONDecodeError:
                    skipped_no_message += 1
                    continue
                if not isinstance(parsed, dict):
                    skipped_no_message += 1
                    continue
                message = parsed.get("message")
                if not isinstance(message, str):
                    skipped_no_message += 1
                    continue

                annotations = openai_reviewer.extract_response_annotations(response)
                metadata = openai_reviewer.extract_response_file_citation_metadata(response)
                rendered = openai_reviewer.render_file_citations_for_markdown(
                    message, annotations, metadata,
                )
                rendered_count += 1

                # No unresolved OpenAI control tokens may survive user-visible
                # output. If a new marker variant escapes the strip regex, this
                # is the assertion that catches it.
                self.assertNotIn("\uE200", rendered)
                self.assertNotIn("\uE201", rendered)
                self.assertNotIn("\uE202", rendered)

                # If the response carried file_citation annotations, the
                # renderer must surface them as a Sources footer in the
                # rendered message.
                has_file_citations = any(
                    isinstance(a, dict)
                    and a.get("type") in ("file_citation", "container_file_citation")
                    for a in annotations
                )
                if has_file_citations:
                    self.assertIn("Sources:", rendered)

                # Prose-survival guard: if a fixture declares a list of
                # ``must_survive`` substrings, every one of them must appear
                # verbatim in the rendered output. This catches overstripping
                # bugs where the marker-strip regex (or its successor) eats
                # adjacent prose -- a class of bug that is silent under the
                # PUA-leak and Sources-footer assertions because the rendered
                # output still has no PUA chars and the footer still renders;
                # only the user-visible prose is missing.
                must_survive = fixture.get("must_survive") or []
                if isinstance(must_survive, list):
                    for chunk in must_survive:
                        if isinstance(chunk, str) and chunk:
                            self.assertIn(
                                chunk, rendered,
                                f"prose chunk {chunk!r} was overstripped from rendered output",
                            )

        # Sanity: if we have fixtures at all, at least some must have been
        # renderable. A run where every fixture is skipped likely indicates a
        # collector or extractor regression rather than a clean tree.
        self.assertGreater(
            rendered_count, 0,
            f"No fixtures produced a renderable message "
            f"(skipped_no_message={skipped_no_message}); "
            "the collector or extract_response_text may be broken.",
        )


if __name__ == "__main__":
    unittest.main()
