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

pr_review_wrapper options: the long tail of flags nothing pinned.

``parse_args()`` reads ``sys.argv`` directly (no argv parameter), so
every case here patches it. Each class names the production reader of
the attribute it pins, because the risk these tests cover is a flag
that parses fine and is then read under a different name -- the
wrapper's namespace travels far (openai_reviewer, podman_repos,
openai_container, openai_vector_store) and nothing else would notice.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_container  # noqa: E402
import openai_vector_store  # noqa: E402
import podman_repos  # noqa: E402
import pr_review_wrapper as wrapper  # noqa: E402

BASE = ["--model", "openai:gpt-5.4"]


def parse(*flags: str) -> argparse.Namespace:
    with mock.patch.object(wrapper.sys, "argv",
                           ["pr_review_wrapper.py", *BASE, *flags]):
        return wrapper.parse_args()


class ModelEnsembleOptionTests(unittest.TestCase):
    """--extra-model / --combine-model build the reviewer ensemble in
    main() and are validated against --codex-host at parse time."""

    def test_extra_model_without_combine_model_fails_at_parse_time(self) -> None:
        """Discovered at run time it would bill every reviewer first."""
        with self.assertRaises(SystemExit) as ctx:
            parse("--extra-model", "anthropic:claude-opus-4")
        self.assertEqual(ctx.exception.code, 2)

    def test_extra_model_is_repeatable_and_ordered(self) -> None:
        args = parse("--extra-model", "anthropic:claude-opus-4",
                     "--extra-model", "zai:glm-5.3",
                     "--combine-model", "openai:gpt-5.4")
        self.assertEqual(args.extra_model,
                         ["anthropic:claude-opus-4", "zai:glm-5.3"])
        self.assertEqual(args.combine_model, "openai:gpt-5.4")

    def test_no_ensemble_by_default(self) -> None:
        args = parse()
        self.assertEqual(args.extra_model, [])
        self.assertIsNone(args.combine_model)

    def test_a_codex_extra_model_needs_a_codex_host(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--extra-model", "codex:gpt-5.6-sol",
                  "--combine-model", "openai:gpt-5.4")
        args = parse("--extra-model", "codex:gpt-5.6-sol",
                     "--combine-model", "openai:gpt-5.4",
                     "--codex-host", "fairy@codexbox")
        self.assertEqual(args.extra_model, ["codex:gpt-5.6-sol"])

    def test_a_codex_combiner_needs_a_codex_host(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--combine-model", "codex:gpt-5.6-sol")


class MainPassPromptTests(unittest.TestCase):
    """The optional PROMPT= on --model / --extra-model. main() reads
    args.main_prompts positionally against [model, *extra_model] to give
    each reviewer its role; everything else must still see a bare spec."""

    def test_no_prefix_leaves_the_prompt_to_the_task(self) -> None:
        args = parse("--extra-model", "zai:glm-5.3",
                     "--combine-model", "openai:gpt-5.4")
        self.assertEqual([None, None], args.main_prompts)

    def test_prefix_is_split_off_and_kept_in_order(self) -> None:
        with mock.patch.object(wrapper.sys, "argv", [
            "pr_review_wrapper.py",
            "--model", "code_review=openai:gpt-5.4",
            "--extra-model", "design_review=zai:glm-5.3@high",
            "--extra-model", "code_review=openai:gpt-5.4",
            "--combine-model", "openai:gpt-5.4",
        ]):
            args = wrapper.parse_args()
        self.assertEqual(["code_review", "design_review", "code_review"],
                         args.main_prompts)
        self.assertEqual("openai:gpt-5.4", args.model)
        self.assertEqual(["zai:glm-5.3@high", "openai:gpt-5.4"], args.extra_model)

    def test_unknown_prompt_is_a_cli_error(self) -> None:
        """Caught here it costs nothing; caught at review time it has
        already billed the reviewers that parsed."""
        with self.assertRaises(SystemExit) as ctx:
            parse("--model", "code_reviw=openai:gpt-5.4")
        self.assertEqual(ctx.exception.code, 2)

    def test_the_other_model_flags_take_no_prompt(self) -> None:
        for flag in ("--triage-model", "--combine-model", "--allowed-model"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parse(flag, "code_review=openai:gpt-5.4")

    def test_an_issue_run_takes_no_prompt(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--task", "issue", "--model", "code_review=openai:gpt-5.4")

    def test_a_prefixed_codex_model_still_needs_a_codex_host(self) -> None:
        """The provider scans read args.model after the split; a prefix
        left on it would silently disable them."""
        with self.assertRaises(SystemExit):
            parse("--model", "code_review=codex:gpt-5.6-sol")
        args = parse("--model", "code_review=codex:gpt-5.6-sol",
                     "--codex-host", "fairy@codexbox")
        self.assertEqual("codex:gpt-5.6-sol", args.model)


class ReviewBudgetOptionTests(unittest.TestCase):
    """--max-output-tokens and --top-p are read by openai_reviewer when
    it builds the Responses call; --openai-timeout-seconds bounds the
    HTTP client main() constructs."""

    def test_max_output_tokens_overrides_the_default(self) -> None:
        self.assertEqual(parse("--max-output-tokens", "12345").max_output_tokens,
                         12345)
        self.assertEqual(parse().max_output_tokens,
                         wrapper.DEFAULT_MAX_OUTPUT_TOKENS)

    def test_top_p_is_a_float_and_unset_by_default(self) -> None:
        self.assertEqual(parse("--top-p", "0.4").top_p, 0.4)
        self.assertIsNone(parse().top_p)

    def test_final_verbosity_defaults_to_unset(self) -> None:
        # None makes main() fall back to --verbosity, so the split is
        # invisible until an operator asks for it.
        args = parse()
        self.assertEqual("high", args.verbosity)
        self.assertIsNone(args.final_verbosity)

    def test_final_verbosity_accepts_the_verbosity_levels(self) -> None:
        args = parse("--verbosity", "low", "--final-verbosity", "medium")
        self.assertEqual("low", args.verbosity)
        self.assertEqual("medium", args.final_verbosity)
        with self.assertRaises(SystemExit):
            parse("--final-verbosity", "verbose")

    def test_openai_timeout_seconds_is_a_float(self) -> None:
        self.assertEqual(parse("--openai-timeout-seconds", "900.5")
                         .openai_timeout_seconds, 900.5)
        self.assertEqual(parse().openai_timeout_seconds,
                         wrapper.DEFAULT_OPENAI_TIMEOUT_SECONDS)


class TriageOptionTests(unittest.TestCase):
    """The triage reviewer's own effort/token budget, read where main()
    builds the triage role."""

    def test_reasoning_effort_accepts_the_documented_levels(self) -> None:
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort):
                self.assertEqual(
                    parse("--triage-reasoning-effort", effort)
                    .triage_reasoning_effort, effort)

    def test_an_unknown_effort_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--triage-reasoning-effort", "extreme")

    def test_defaults_come_from_the_module_constants(self) -> None:
        args = parse()
        self.assertEqual(args.triage_reasoning_effort,
                         wrapper.DEFAULT_TRIAGE_REASONING_EFFORT)
        self.assertEqual(args.triage_max_output_tokens,
                         wrapper.DEFAULT_TRIAGE_MAX_OUTPUT_TOKENS)

    def test_triage_max_output_tokens_overrides_the_default(self) -> None:
        self.assertEqual(parse("--triage-max-output-tokens", "2000")
                         .triage_max_output_tokens, 2000)


class ContainerOptionTests(unittest.TestCase):
    """Shape of the auto-created OpenAI container, read where main()
    calls into openai_container."""

    def test_expiry_minutes_overrides_the_default(self) -> None:
        self.assertEqual(parse("--container-expiry-minutes", "45")
                         .container_expiry_minutes, 45)
        self.assertEqual(parse().container_expiry_minutes,
                         openai_container.DEFAULT_CONTAINER_EXPIRY_MINUTES)

    def test_memory_limit_accepts_only_the_offered_sizes(self) -> None:
        self.assertEqual(parse("--container-memory-limit", "16g")
                         .container_memory_limit, "16g")
        self.assertEqual(parse().container_memory_limit,
                         openai_container.DEFAULT_CONTAINER_MEMORY_LIMIT)
        with self.assertRaises(SystemExit):
            parse("--container-memory-limit", "128g")


class VectorStoreOptionTests(unittest.TestCase):
    """Vector-store lifetime knobs, read where main() syncs the store."""

    def test_expiry_days_and_retries_override_the_defaults(self) -> None:
        args = parse("--vector-store-expiry-days", "3",
                     "--vector-store-sync-max-retries", "7")
        self.assertEqual(args.vector_store_expiry_days, 3)
        self.assertEqual(args.vector_store_sync_max_retries, 7)

    def test_defaults_come_from_the_vector_store_module(self) -> None:
        args = parse()
        self.assertEqual(args.vector_store_expiry_days,
                         openai_vector_store.DEFAULT_VECTOR_STORE_EXPIRY_DAYS)
        self.assertEqual(args.vector_store_sync_max_retries,
                         openai_vector_store.DEFAULT_VECTOR_STORE_SYNC_MAX_RETRIES)

    def test_prepare_only_is_off_unless_asked(self) -> None:
        self.assertTrue(parse("--prepare-vector-store-only")
                        .prepare_vector_store_only)
        self.assertFalse(parse().prepare_vector_store_only)


class PodmanOptionTests(unittest.TestCase):
    """--podman-ssh-identity is consumed at parse time (it becomes part
    of every parsed host); --podman-mirror-root reaches podman_repos and
    --podman-parallel-tool-calls reaches openai_reviewer."""

    def test_the_identity_rides_on_every_parsed_shell_host(self) -> None:
        args = parse("--shell-host", "fairy@h1", "--shell-host", "arm64=fairy@h2",
                     "--codex-host", "fairy@codexbox",
                     "--podman-ssh-identity", "/keys/fairy_ed25519")
        self.assertEqual([m.host.identity for m in args.machines],
                         ["/keys/fairy_ed25519"] * 2)
        self.assertEqual(args.codex_host.host.identity, "/keys/fairy_ed25519")

    def test_without_it_ssh_picks_the_key_itself(self) -> None:
        args = parse("--shell-host", "fairy@h1")
        self.assertIsNone(args.machines[0].host.identity)

    def test_mirror_root_overrides_the_podman_repos_default(self) -> None:
        self.assertEqual(parse("--podman-mirror-root", "srv/mirrors")
                         .podman_mirror_root, "srv/mirrors")
        self.assertEqual(parse().podman_mirror_root,
                         podman_repos.DEFAULT_MIRROR_ROOT)

    def test_parallel_tool_calls_is_off_unless_asked(self) -> None:
        self.assertTrue(parse("--podman-parallel-tool-calls")
                        .podman_parallel_tool_calls)
        self.assertFalse(parse().podman_parallel_tool_calls)

    def test_podman_without_a_shell_host_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--podman")

    def test_duplicate_shell_host_labels_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--shell-host", "big=fairy@h1", "--shell-host", "big=fairy@h2")


class WebSearchDomainOptionTests(unittest.TestCase):
    """--web-search-domain reaches the reviewer's web_search tool."""

    def test_it_is_repeatable_and_empty_by_default(self) -> None:
        args = parse("--web-search-domain", "ffmpeg.org",
                     "--web-search-domain", "trac.ffmpeg.org")
        self.assertEqual(args.web_search_domain,
                         ["ffmpeg.org", "trac.ffmpeg.org"])
        self.assertEqual(parse().web_search_domain, [])


class ConcurrencyOptionTests(unittest.TestCase):
    """--concurrency is parsed into (provider, count) pairs that main()
    hands to concurrency.configure."""

    def test_limits_parse_into_provider_count_pairs(self) -> None:
        args = parse("--concurrency", "codex:2", "--concurrency", "openai:4")
        self.assertEqual(args.concurrency, [("codex", 2), ("openai", 4)])

    def test_no_limits_by_default(self) -> None:
        self.assertEqual(parse().concurrency, [])

    def test_a_malformed_limit_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse("--concurrency", "codex")


class PersistBranchesOptionTests(unittest.TestCase):
    """--persist-branches is validated against --podman at parse time."""

    def test_requires_podman(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            parse("--persist-branches")
        self.assertEqual(ctx.exception.code, 2)

    def test_accepted_for_the_issue_task_too(self) -> None:
        ns = parse("--persist-branches", "--task", "issue",
                   "--podman", "--shell-host", "fairy@203.0.113.7")
        self.assertTrue(ns.persist_branches)

    def test_accepted_with_podman_and_repos(self) -> None:
        ns = parse("--persist-branches", "--podman",
                   "--shell-host", "fairy@203.0.113.7",
                   "--persist-repo", "ffmpeg",
                   "--persist-repo", "ffmpeg-web")
        self.assertTrue(ns.persist_branches)
        self.assertEqual(ns.persist_repo, ["ffmpeg", "ffmpeg-web"])


if __name__ == "__main__":
    unittest.main()
