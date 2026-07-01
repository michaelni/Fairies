"""Tests for the ``flex`` -> ``default`` last-attempt fallback.

Today's failure mode (observed 2026-05-07 in fairy.log against
production OpenAI): the ``flex`` service tier pool gets saturated
and ``responses.create`` returns 429 / Cloudflare 502 indefinitely
while the wrapper's rate-limit retry loop spins until the outer
``--llm-timeout`` fires. With ``--llm-max-attempts 3`` that wastes
up to ``3 * llm_timeout`` per PR on the same dead-end queue.

``flex_fallback_extra_args`` extracts the last-attempt fallback
policy: when (and only when) the configured wrapper cmd actually
asked for ``--service-tier flex`` and/or ``--triage-service-tier
flex``, the helper returns the argparse-last-wins overrides that
downgrade those tiers to ``default``. Other tiers and "tier unset"
are left alone so the helper is a no-op for non-flex setups.

These tests pin the parsing edge cases (``--flag value`` vs
``--flag=value``, both flags at once, only one set, neither set,
non-flex tiers) so future readers don't have to re-derive them
from the loop in ``flex_fallback_extra_args``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


class FlexFallbackExtraArgsTests(unittest.TestCase):
    def test_no_cmd_returns_empty(self) -> None:
        self.assertEqual(fairy.flex_fallback_extra_args(""), [])

    def test_no_service_tier_returns_empty(self) -> None:
        cmd = "./openai_pr_review_wrapper.py --model gpt-5.4 --reasoning-effort high"
        self.assertEqual(fairy.flex_fallback_extra_args(cmd), [])

    def test_non_flex_tier_returns_empty(self) -> None:
        # Operator already on ``default`` / ``priority`` -- nothing to
        # downgrade. The fallback only fires for flex.
        for tier in ("auto", "default", "priority"):
            with self.subTest(tier=tier):
                cmd = f"./wrapper.py --service-tier {tier}"
                self.assertEqual(fairy.flex_fallback_extra_args(cmd), [])

    def test_main_flex_separate_token(self) -> None:
        cmd = "./wrapper.py --model gpt-5.4 --service-tier flex --reasoning-effort high"
        self.assertEqual(
            fairy.flex_fallback_extra_args(cmd),
            ["--service-tier", "default"],
        )

    def test_main_flex_equals_form(self) -> None:
        cmd = "./wrapper.py --service-tier=flex --reasoning-effort high"
        self.assertEqual(
            fairy.flex_fallback_extra_args(cmd),
            ["--service-tier", "default"],
        )

    def test_triage_flex_only(self) -> None:
        # Triage flex without main flex: only the triage flag gets a
        # downgrade. Main tier is left untouched (unset == account
        # default already).
        cmd = "./wrapper.py --triage-service-tier flex --triage-model gpt-5.4-mini"
        self.assertEqual(
            fairy.flex_fallback_extra_args(cmd),
            ["--triage-service-tier", "default"],
        )

    def test_both_flex(self) -> None:
        cmd = (
            "./wrapper.py --service-tier flex "
            "--triage-service-tier=flex --triage-model gpt-5.4-mini"
        )
        self.assertEqual(
            fairy.flex_fallback_extra_args(cmd),
            [
                "--service-tier", "default",
                "--triage-service-tier", "default",
            ],
        )

    def test_real_failing_cmd_from_log(self) -> None:
        # Verbatim from fairy.log 2026-05-07: PR #20997 timed out
        # after 7200s on flex. Exercising the actual cmd shape that
        # motivated the fallback guards against future quoting / flag
        # re-orderings silently regressing the match.
        cmd = (
            "./openai_pr_review_wrapper.py --repo-root ffmpeg --model gpt-5.4 "
            "--triage-model gpt-5.4-mini --extra-repo-root all_ffmpeg "
            "--use-vector-store-search --reasoning-effort high --verbose "
            "--debug-response-dir openaidebug --use-web-search "
            "--use-openai-container-repos --max-tool-calls 100 "
            "--service-tier flex --allowed-model gpt-5.5 --allowed-model gpt-5.4"
        )
        self.assertEqual(
            fairy.flex_fallback_extra_args(cmd),
            ["--service-tier", "default"],
        )

    def test_flex_substring_in_other_value_is_ignored(self) -> None:
        # ``flex`` appearing as part of another flag's value (not as
        # the value of a service-tier flag) must not trigger the
        # fallback. This guards against a naive ``"flex" in cmd``
        # check sneaking in during a refactor.
        cmd = "./wrapper.py --debug-response-dir openaidebug-flex --service-tier default"
        self.assertEqual(fairy.flex_fallback_extra_args(cmd), [])


if __name__ == "__main__":
    unittest.main()
