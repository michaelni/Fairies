#!/usr/bin/env python3
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

Auto-approve stale Forgejo/Gitea pull requests via gcli.

Selection rules:
1. PR is open.
2. PR is not marked WIP/draft.
3. PR has no *currently outstanding* request-for-changes review.
4. The PR head commit has CI statuses and every latest reported context is successful
   (unless ``--triage-on-ci-failure`` is set together with a triage-capable
   ``--llm-review-cmd``; then ERROR/FAILURE jobs can be passed to the mini
   model for a short heads-up, with deduplication so fairy does not re-post
   when its prior comments already mention every failing context).
5. PR has had no activity for at least 7 days.

Optional LLM review:
When --llm-review-cmd is set, PRs that would otherwise be approved are sent to an
external reviewer command. That command receives JSON on stdin containing a fixed
review prompt, PR metadata, and the patch text. It must return JSON with one of
these classifications:

- approve
- minor_issues_approve
- moderate_issues
- major_issues
- skip

The script uses `gcli api` for reads and `gcli pulls ... approve` for approval.
By default it runs in dry-run mode. Pass --approve to actually submit approvals,
or --manual to confirm actions one by one interactively.

When a human @-mentions fairy or requests it as a reviewer, the run may enter a
"forced review" path: the PR is *never* auto-approved in that case unless
``--llm-review-cmd`` is set; without an LLM command the PR is simply skipped
(no approval).
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import logging
import os
import re
from queue import Empty, SimpleQueue
from threading import Lock, Thread
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace as dataclasses_replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Protocol, TypeAlias
from urllib.parse import quote, urlencode, urljoin

import ci_log
import git_util
import gcli_cache
from common import (
    JsonObject,
    add_color_arg,
    apply_config_file_defaults,
    attachment_urls,
    default_cache_path,
    iso_to_dt,
    parse_iso_datetime_arg,
    setup_logging,
)
import forge_gcli
from forge_gcli import (
    add_forge_repo_args,
    apply_issue_label_changes,
    build_repo_path,
    gcli_api,
    gcli_prefix,
    list_issue_timeline,
    load_json,
    post_issue_comment,
    run_cmd,
)
from forgejo_export import labels
import workset


DEFAULT_WIP_PREFIXES = ["WIP:", "[WIP]"]

# Once fairy has already reviewed a PR, a later push only needs to settle
# for 6h (not the full --min-age-days) before she re-reviews.
REVIEWED_PR_MIN_AGE_DAYS = 0.25

ANSI_RED = "\033[31m"
ANSI_BOLD_RED = "\033[1;91m"
ANSI_RESET = "\033[0m"

ACTIONABLE_DECISIONS = frozenset({"approve", "comment", "request_changes"})


@dataclass(frozen=True)
class LabelChange:
    """One add/remove of a single label proposed by the triager.

    ``reason`` justifies the change; ``post`` is True when that reason is
    needed to understand the label and should be posted to the PR as a
    comment, False when it only serves logs.
    """
    label: str
    op: str  # "add" | "remove"
    reason: str = ""
    post: bool = False


@dataclass(frozen=True)
class LLMReview:
    classification: str
    message: str
    label_changes: tuple[LabelChange, ...] = ()


@dataclass(frozen=True)
class Decision:
    pr_number: int
    title: str
    author: str
    auto_merge: str
    action: str
    reason: str
    last_activity: datetime | None
    llm_classification: str = "-"
    llm_message: str = ""
    expected_pr_updated_at: str | None = None
    expected_head_ref: str | None = None
    # Names of commit-status contexts whose latest state needs a
    # human to act in the Forgejo UI. Both fields are propagated to
    # the end-of-run summary which lists affected PRs grouped by
    # state; fairy has no API to retry / unblock either. Empty
    # tuple for all other PRs.
    cancelled_ci_contexts: tuple[str, ...] = ()
    blocked_ci_contexts: tuple[str, ...] = ()
    # True iff this PR has already been approved by fairy and is
    # neither queued for auto-merge nor blocked by conflicts -- i.e.
    # all fairy can do is wait for a human to click "Merge". The
    # end-of-run summary lists these as a reminder. ``pr.mergeable``
    # is implicitly True because the upstream conflicts gate would
    # otherwise have skipped the PR before we got here.
    merge_ready: bool = False
    # Non-bot reviewers whose latest review is APPROVED, when no one
    # has CHANGES_REQUESTED outstanding and auto-merge is not queued.
    # Captures PRs fairy was never involved with that an external
    # maintainer has already approved -- i.e. the second flavor of
    # "needs a human to click Merge". Empty for the merge_ready case
    # to keep the two summary lists disjoint.
    external_approvers: tuple[str, ...] = ()
    label_changes: tuple[LabelChange, ...] = ()


@dataclass(frozen=True)
class PreparedPR:
    pr: ApiObject
    number: int
    title: str
    author: str
    auto_merge: str
    last_activity: datetime | None
    base_reason: str
    discussion: list[DiscussionItem]
    reviewer_username: str | None
    # When the PR head is CI-red, optional structured payload for the review
    # wrapper so triage can suggest a short heads-up. ``None`` for green CI
    # or for PRs that did not go through the CI-triage path.
    ci_triage: JsonObject | None = None
    # Names of commit-status contexts whose latest state needs a
    # human to act in the Forgejo UI. Propagated into the resulting
    # ``Decision`` so the end-of-run summary can list PRs that need
    # manual rerun / unblock in the UI.
    cancelled_ci_contexts: tuple[str, ...] = ()
    blocked_ci_contexts: tuple[str, ...] = ()
    # Same semantics as on ``Decision``; carried through the LLM path
    # and patched onto the resulting ``Decision`` at the funnel point
    # in ``safe_apply_llm_review_to_prepared``.
    external_approvers: tuple[str, ...] = ()
    # Cross-process contract with the --llm-review-cmd wrapper: when
    # True, the wrapper runs its full reviewer pass even if its triage
    # pre-check votes ``skip`` (see --force-review-skip). Travels as the
    # ``ignore_triage_skip`` field in the wrapper's stdin request.
    ignore_triage_skip: bool = False
    # Stronger sibling of ``ignore_triage_skip`` (see --force-engage):
    # when True, the wrapper engages its full reviewer pass regardless of
    # the triage route (skip / reply_no_verdict) and even on CI-red PRs.
    # Travels as the ``force_engage`` field in the wrapper's stdin request.
    force_engage: bool = False


logger = logging.getLogger(__name__)


ApiObject: TypeAlias = dict[str, object]
DiscussionItem: TypeAlias = dict[str, object]
ActivityPredicate: TypeAlias = Callable[[ApiObject], bool]
PreparedItem: TypeAlias = Decision | PreparedPR

_LLM_REVIEW_DONE = object()
_PREPARE_DONE = object()


def parse_pr_number_csv(value: str) -> list[int]:
    numbers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid PR number: {part!r}") from exc
        if number <= 0:
            raise argparse.ArgumentTypeError(f"invalid PR number: {part!r}")
        numbers.append(number)
    if not numbers:
        raise argparse.ArgumentTypeError("expected one or more PR numbers")
    return numbers


def parse_label_csv(value: str) -> list[str]:
    labels = [p.strip() for p in value.split(",") if p.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("expected one or more comma-separated label names")
    return labels


def flatten_pr_number_args(values: list[list[int]] | None) -> set[int]:
    numbers: set[int] = set()
    for group in values or []:
        numbers.update(group)
    return numbers


def flatten_label_args(values: list[list[str]] | None) -> list[str]:
    return list(dict.fromkeys(
        label for group in (values or []) for label in group
    ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Approve stale Forgejo/Gitea PRs via gcli.",
    )
    add_forge_repo_args(p)
    p.add_argument(
        "--min-age-days",
        type=float,
        default=7.0,
        help=(
            "Minimum age of last *discussion* activity (in days) before fairy will "
            "proactively use the LLM, including the optional CI-failure triage path "
            "(--triage-on-ci-failure). This does not apply when a human @-mentions the "
            "bot or sets fairy as a requested reviewer (fairy is expected to "
            "answer promptly in those cases; see forced-review logic). "
            "If fairy has posted any prior review, the effective threshold is at "
            "most 1.0 day (same as before). Default: 7."
        ),
    )
    p.add_argument(
        "--approve",
        action="store_true",
        help="Actually submit matching reviews/comments. Without this flag the script only reports matches.",
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help="Interactively confirm each matching action.",
    )
    p.add_argument(
        "--approve-message",
        default="",
        help="Optional approval message body.",
    )
    p.add_argument(
        "--include-self-approved",
        action="store_true",
        help="Approve even if your latest effective review state is already APPROVED.",
    )
    p.add_argument(
        "--wip-prefix",
        action="append",
        dest="wip_prefixes",
        default=None,
        help=(
            "Additional WIP title prefix. Can be repeated. "
            "Defaults to the common Forgejo/Gitea prefixes WIP: and [WIP]."
        ),
    )
    p.add_argument(
        "--llm-review-cmd",
        help=(
            "External command used to review candidate PRs. It receives JSON on stdin "
            "and must print JSON with classification and message on stdout."
        ),
    )
    p.add_argument(
        "--podman-host",
        action="append",
        default=[],
        metavar="[LABEL=]USER@HOST[,port=N][,cpus=N][,memory=SIZE][,gpu=DEV]",
        help=(
            "Run LLM shell work (review, triage, ...) in ephemeral "
            "containers on this podman host (passwordless ssh destination); "
            "repeat for more machines, the first being the default. Each "
            "value is forwarded as --shell-host to --llm-review-cmd, so that "
            "command must be the openai wrapper. Provision each host first "
            "with containers/provision_remote.py."
        ),
    )
    p.add_argument(
        "--codex-host",
        default=None,
        metavar="[LABEL=]USER@HOST",
        help=(
            "podman host that runs the codex container, forwarded as "
            "--codex-host to --llm-review-cmd. Required for any codex: model "
            "spec (codex runs only in a container there). Provision it with "
            "containers/provision_remote.py --codex-bin."
        ),
    )
    p.add_argument(
        "--codex-home",
        default=None,
        metavar="DIR",
        help=(
            "Wrapper-side CODEX_HOME (the bot's `codex login`), forwarded as "
            "--codex-home to --llm-review-cmd. Default: the wrapper's "
            "environment / codex's ~/.codex."
        ),
    )
    p.add_argument(
        "--llm-timeout",
        type=int,
        default=3600*5,
        help="Timeout in seconds for the external LLM review command (default: 14400)",
    )
    p.add_argument(
        "--llm-parallelism",
        type=int,
        default=1,
        help=(
            "Number of LLM review subprocesses to run in parallel (default: 1). "
            "Requires the underlying review command to be safe for concurrent "
            "invocation. Higher values shorten wall time on PR batches at the cost "
            "of proportionally higher OpenAI/API request rates."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N LLM evaluations (0 = no limit). Caps cost and "
             "provider rate-limit windows; gate-skipped PRs do not count.",
    )
    p.add_argument(
        "--llm-max-attempts",
        type=int,
        default=3,
        help=(
            "Maximum number of LLM review attempts per PR (default: 3). "
            "If the LLM review command fails (non-zero exit, timeout, malformed "
            "JSON, or unknown classification) we retry up to this many total "
            "attempts before giving up and recording the PR as 'error'."
        ),
    )
    p.add_argument(
        "--llm-retry-delay",
        type=float,
        default=5.0,
        help=(
            "Seconds to sleep between failed LLM review attempts (default: 5). "
            "Applied only between attempts; no delay before the first attempt "
            "or after the final one."
        ),
    )
    p.add_argument(
        "--triage-on-ci-failure",
        action="store_true",
        help=(
            "When the head commit has failing CI jobs (ERROR/FAILURE), do not stop "
            "immediately. If your LLM review command is configured for triage "
            "(e.g. pr_review_wrapper.py with --triage-model), run it so the "
            "mini model can post a short reply_no_verdict pointing at the failure. "
            "If every failing context was already mentioned in a prior comment by the "
            "bot, the wrapper is not invoked. Requires --llm-review-cmd and a "
            "command line that includes --triage-model."
        ),
    )
    p.add_argument(
        "--ci-failure-log-lines",
        type=int,
        default=300,
        metavar="N",
        help=(
            "When a triage or reviewer LLM is about to run on a PR with failing "
            "CI, attach the last N lines of each failing job's log to the "
            "ci_triage payload so the model sees the real error, not just the "
            "one-line status. 0 disables the log fetch (default: 300)."
        ),
    )
    p.add_argument(
        "--triage-label",
        action="append",
        type=parse_label_csv,
        dest="triage_label",
        default=None,
        metavar="LABEL[,LABEL...]",
        help=(
            "Label name the triage model may add or remove. Can be repeated "
            "or passed as a comma-separated list. Passed to the LLM review "
            "command as ``triage_label_allowlist`` in the stdin JSON payload."
        ),
    )
    p.add_argument(
        "--llm-max-patch-bytes",
        type=int,
        default=200000,
        help="Maximum number of patch bytes sent to the LLM (default: 200000)",
    )
    p.add_argument(
        "--patch-repo",
        type=Path,
        metavar="PATH",
        help="Local clone used to synthesize the PR patch via git format-patch (required with --llm-review-cmd).",
    )
    p.add_argument(
        "--patch-pr-ref-template",
        metavar="TEMPLATE",
        help="Ref pattern resolving a PR's branch in --patch-repo, with "
             "``{number}`` substituted (e.g. ``fforge/pr/{number}``). "
             "Required with --simulate-past so each PR's historical head "
             "comes from refs the operator pinned in the prepped mirror.",
    )
    p.add_argument(
        "--force-review-pr",
        action="append",
        type=parse_pr_number_csv,
        default=None,
        metavar="N[,N...]",
        help=(
            "Force review and potential approval for the specified PR number(s), bypassing the usual "
            "selection checks. Can be repeated or passed as a comma-separated list."
        ),
    )
    p.add_argument(
        "--force-skip-pr",
        action="append",
        type=parse_pr_number_csv,
        default=None,
        metavar="N[,N...]",
        help=(
            "Always skip review and approval for the specified PR number(s). Can be repeated or passed "
            "as a comma-separated list. Takes precedence over --force-review-pr."
        ),
    )
    p.add_argument(
        "--forced-only",
        action="store_true",
        help="Limit the run to PRs named via --force-review-pr "
             "(no open-PR listing). Requires at least one --force-review-pr.",
    )
    p.add_argument(
        "--force-review-non-open",
        action="store_true",
        help="Let --force-review-pr also review closed/merged PRs. Off by "
             "default: a forced PR whose state is not ``open`` is skipped "
             "with ``not open``. (WIP/draft and conflicting PRs are always "
             "reviewed when forced, independent of this flag.)",
    )
    p.add_argument(
        "--force-review-skip",
        action="store_true",
        help="When reviewing a PR named via --force-review-pr, ignore a "
             "``skip`` verdict from the --llm-review-cmd triage pre-check and "
             "run the full reviewer pass anyway. Only applies to PRs named "
             "with --force-review-pr (not @mention / requested-reviewer "
             "engagements).",
    )
    p.add_argument(
        "--force-engage",
        action="store_true",
        help="When reviewing a PR named via --force-review-pr, run the full "
             "reviewer pass regardless of the triage route (overrides both "
             "``skip`` and ``reply_no_verdict``) and even when the head CI is red. "
             "Stronger than --force-review-skip. Only applies to PRs named with "
             "--force-review-pr (not @mention / requested-reviewer engagements).",
    )
    p.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Verbosity level for command logging: "
            "0 = none, "
            "1 = write commands and initial PR list, "
            "2 = all commands."
        ),
    )
    add_color_arg(p)
    p.add_argument(
        "--simulate-past",
        type=parse_iso_datetime_arg,
        metavar="ISO_DATETIME",
        help="Filter cached events to ISO_DATETIME and resolve PR heads "
             "from --patch-repo refs (which the operator preps for the "
             "cutoff). Pair with --patch-pr-ref-template and a separate "
             "--cache. See startup warning for residual limits.",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=default_cache_path("pr_data_cache.pkl"),
        help="Pickle cache path holding per-PR gcli data "
             "(default: ~/.fairy/pr_data_cache.pkl).",
    )
    p.add_argument(
        "--workset-dir",
        type=Path,
        default=default_cache_path("workset"),
        help="Root of the persistent per-item JSON work files "
             "(default: ~/.fairy/workset).",
    )
    p.add_argument(
        "--workset-retention-days",
        type=float,
        default=14.0,
        help="Days after an item leaves the open listing before its "
             "finished workset file is deleted (default: 14).",
    )
    p.add_argument(
        "--discussion-cache-max-age-hours",
        type=float,
        default=24.0,
        help="Time-to-live (hours) on the cached comments / reviews "
             "/ inline review-comments trio. These three fields can be "
             "silently edited or deleted server-side without bumping "
             "pr.updated_at; the TTL forces a periodic refetch as a "
             "backstop. Other PR fields (timeline, commits, files) are "
             "gated only on pr.updated_at and ignore this TTL. "
             "(default: 24)",
    )
    apply_config_file_defaults(p, argv)
    args = p.parse_args(argv)
    args.force_review_prs = flatten_pr_number_args(args.force_review_pr)
    args.force_skip_prs = flatten_pr_number_args(args.force_skip_pr)
    args.triage_labels = flatten_label_args(args.triage_label)
    return args


def combine_review_messages(*parts: str) -> str:
    clean = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(clean)


def gcli_approve(args: argparse.Namespace, pr_number: int, review_message: str = "") -> None:
    cmd = gcli_prefix(args) + [
        "pulls",
        "-o",
        args.owner,
        "-r",
        args.repo,
        "-i",
        str(pr_number),
        "approve",
        "-y",
    ]

    tmp_path: str | None = None
    try:
        message = combine_review_messages(args.approve_message, review_message)
        if message:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
                tf.write(message)
                tmp_path = tf.name
            cmd += ["-T", tmp_path]

        cp = run_cmd(cmd, verbose=args.verbose)
        if cp.returncode != 0:
            raise RuntimeError(
                f"gcli approve failed for PR #{pr_number} with exit code {cp.returncode}:\n"
                f"{cp.stderr.strip()}"
            )
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def get_submission_guard(prepared: PreparedItem, decision: Decision) -> tuple[str | None, str | None]:
    if isinstance(prepared, PreparedPR):
        return prepared.pr.get("updated_at"), get_pr_head_ref(prepared.pr)
    return decision.expected_pr_updated_at, decision.expected_head_ref


def check_pr_still_unchanged(
    args: argparse.Namespace,
    prepared: PreparedItem,
    decision: Decision,
) -> str | None:
    expected_pr_updated_at, expected_head_ref = get_submission_guard(prepared, decision)
    current = get_pr(args, decision.pr_number)
    # ``--force-review-pr`` with --force-review-non-open posts even to a
    # closed/merged PR (the forge still accepts comments there); the
    # heuristic open-state guard is for cron mode. Without the opt-in,
    # a PR that closed between prepare and post is left alone, matching
    # the prepare-time gate. The updated_at/head staleness checks below
    # stay active so a forced post is still pinned to the reviewed state.
    forced_non_open = (
        decision.pr_number in args.force_review_prs and args.force_review_non_open
    )
    if current.get("state") != "open" and not forced_non_open:
        return "PR is no longer open"
    if expected_pr_updated_at is not None and current.get("updated_at") != expected_pr_updated_at:
        return "PR updated_at changed"
    if expected_head_ref is not None and get_pr_head_ref(current) != expected_head_ref:
        return "PR head changed"
    return None


def submit_decision_action(
    args: argparse.Namespace,
    prepared: PreparedItem,
    decision: Decision,
    *,
    cache: gcli_cache.Cache | None = None,
) -> bool:
    changed_reason = check_pr_still_unchanged(args, prepared, decision)
    if changed_reason is not None:
        logger.info("PR #%s: SKIP            submit skipped because %s", decision.pr_number, changed_reason)
        return False
    if decision.action == "approve":
        gcli_approve(args, decision.pr_number, decision.llm_message)
    elif decision.action == "comment" or decision.action == "request_changes":
        post_issue_comment(
            args, args.owner, args.repo, decision.pr_number, decision.llm_message,
        )
    else:
        raise RuntimeError(f"unexpected actionable decision: {decision.action!r}")

    # Invalidate the cached PR entry after a successful post.
    #
    # Without this, the next run can hit a stale-cache race: the in-memory
    # entry written earlier in the current run recorded the pre-post
    # ``issue_comments`` / ``reviews`` under the latest ``pr.updated_at``.
    # Forgejo's ``/pulls/N`` and ``/issues/N/comments`` endpoints are
    # eventually consistent; after our write both may eventually reflect the
    # new comment, but there is a window where ``pr.updated_at`` matches the
    # value we already cached while the comments list still lacks our
    # comment. The cache's freshness check is keyed on ``updated_at``
    # equality, so it would accept the stale comments list indefinitely.
    # Dropping the entry forces a full refetch next time, at which point
    # Forgejo has had time to propagate the new comment.
    #
    # Persist the pop immediately so it survives an unclean kill of this
    # run before the outer ``finally``-block save has a chance to execute.
    if cache is not None:
        cache.entries.pop(
            gcli_cache.EntryKey("pulls", args.owner, args.repo, decision.pr_number),
            None,
        )
        try:
            gcli_cache.save_cache(args.cache, cache)
        except Exception as exc:
            logger.warning(
                "failed to persist cache invalidation after posting %s "
                "for PR #%s: %s",
                decision.action, decision.pr_number, exc,
            )
    return True


def first_dt(obj: ApiObject, *keys: str) -> datetime | None:
    for key in keys:
        dt = iso_to_dt(obj.get(key))
        if dt is not None:
            return dt
    return None


def max_dt(values: Iterable[datetime | None]) -> datetime | None:
    return max((v for v in values if v is not None), default=None)


def list_open_prs(args: argparse.Namespace) -> list[ApiObject]:
    query = urlencode({"state": "open", "sort": "leastupdate", "limit": 100})
    path = build_repo_path(args.owner, args.repo, f"/pulls?{query}")
    data = gcli_api(args, path, all_pages=True, verbose_threshold=1)
    if not isinstance(data, list):
        raise RuntimeError(f"expected list of PRs, got {type(data).__name__}")
    return [pr for pr in data if isinstance(pr, dict)]


def get_pr(args: argparse.Namespace, pr_number: int) -> ApiObject:
    path = build_repo_path(args.owner, args.repo, f"/pulls/{pr_number}")
    data = gcli_api(args, path)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected PR object for #{pr_number}, got {type(data).__name__}")
    return data


def get_pr_discussion(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    cache: gcli_cache.Cache,
    cache_max_age: timedelta,
) -> tuple[list[ApiObject], list[ApiObject], list[ApiObject]]:
    """Return ``(reviews, comments, review_comments)`` for ``pr`` via the cache.

    Thin adapter over ``gcli_cache.get`` kept as a named
    call site so the LLM-decision path reads "fetch the discussion"
    instead of spelling out the field triple. The cache module owns
    the freshness logic (``updated_at``-keyed plus a TTL on the
    edit-prone trio); this function only converts the returned
    tuples to lists for the call-site shape.
    """
    pr_number = int(pr["number"])
    live_updated_at = iso_to_dt(pr.get("updated_at"))
    if live_updated_at is None:
        raise RuntimeError(
            f"PR #{pr_number} is missing or has unparseable updated_at; "
            f"refusing to cache against an unknown freshness key"
        )
    fields = gcli_cache.get(
        cache, args, "pulls", args.owner, args.repo, pr_number, live_updated_at,
        "reviews", "issue_comments", "review_comments",
        max_age=cache_max_age,
    )
    return (
        list(fields["reviews"]),
        list(fields["issue_comments"]),
        list(fields["review_comments"]),
    )


def list_commit_statuses(args: argparse.Namespace, ref: str) -> list[ApiObject]:
    path = build_repo_path(args.owner, args.repo, f"/commits/{quote(ref, safe='')}/statuses")
    data = gcli_api(args, path, all_pages=True)
    if not isinstance(data, list):
        return []
    statuses = [s for s in data if isinstance(s, dict)]
    return filter_activity_after(
        statuses, getattr(args, "simulate_past", None), "created_at", "updated_at",
    )


def get_self_login(args: argparse.Namespace) -> str | None:
    try:
        data = gcli_api(args, "/user", all_pages=False)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("login", "username"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def normalize_review_state(state: str | None) -> str | None:
    if not state:
        return None
    s = state.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "APPROVE": "APPROVED",
        "APPROVED": "APPROVED",
        "REQUEST_CHANGES": "CHANGES_REQUESTED",
        "REQUESTED_CHANGES": "CHANGES_REQUESTED",
        "CHANGES_REQUESTED": "CHANGES_REQUESTED",
        "REJECTED": "CHANGES_REQUESTED",
        "COMMENT": "COMMENTED",
        "COMMENTED": "COMMENTED",
        "PENDING": "PENDING",
    }
    return mapping.get(s, s)



def normalize_status_state(state: str | None) -> str | None:
    """Normalize the ``state``/``status`` field of a single commit-status row.

    This is a pure string normalizer. For Forgejo Actions cancellations
    the row's state field is ``failure`` (not ``cancelled``) and the
    cancellation hint lives in the row's ``description`` instead -- use
    ``row_effective_state(row)`` for that case, which combines this
    normalizer with the description override.

    The ``CANCELLED`` / ``CANCELED`` mapping entries below are real
    (not dead): GitLab's commit-status API emits
    ``state="canceled"`` directly when a pipeline is cancelled, and
    a future Forgejo version may surface ``cancelled`` here too.
    """
    if not state:
        return None
    s = state.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "SUCCESS": "SUCCESS",
        "SUCCESSFUL": "SUCCESS",
        "OK": "SUCCESS",
        "PENDING": "PENDING",
        "RUNNING": "PENDING",
        "IN_PROGRESS": "PENDING",
        "QUEUED": "PENDING",
        "WAITING": "PENDING",
        "NEUTRAL": "NEUTRAL",
        "SKIPPED": "NEUTRAL",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "FAILURE": "FAILURE",
        "FAILED": "FAILURE",
        # ``CANCELLED`` and ``CANCELED`` (single-L, GitLab spelling) are
        # kept as a distinct logical state rather than collapsed into
        # ``FAILURE``: a cancelled job is usually an infra hiccup or a
        # superseded run that just needs a click on the UI's "Rerun"
        # button. Forgejo's HTTP API exposes no rerun endpoint
        # (verified against the swagger spec -- only ``/dispatches``
        # exists for writes), so fairy can't drive the retry itself.
        # If a future Forgejo release adds one, this code can be taught
        # to use it. Instead, callers that build the CI-triage
        # payload for the LLM exclude ``CANCELLED`` contexts (see
        # ``CI_TRIAGE_NAG_STATES``) and the operator-facing summary at
        # end-of-run lists the affected PRs so a human can click the
        # Rerun button. ``TIMED_OUT`` is intentionally still folded
        # into ``FAILURE`` -- a timeout is much more likely to be a
        # real test problem than a transient infra issue, so we still
        # want the LLM to nag about it.
        "CANCELLED": "CANCELLED",
        "CANCELED": "CANCELLED",
        "TIMED_OUT": "FAILURE",
    }
    return mapping.get(s, s)


# Forgejo Actions reports cancelled jobs through the legacy
# commit-status API as ``status="failure"`` with
# ``description="Has been cancelled"`` -- the legacy API has no
# ``cancelled`` state value, so the cancellation signal has to be
# recovered from the description text. Behavior first observed on
# Forgejo 15.0.0+gitea-1.22.0 against
# ``GET /repos/.../commits/<sha>/statuses`` (sample row:
# ``{"status":"failure","description":"Has been cancelled",...}``)
# and expected to be stable across Forgejo releases since it
# follows from the legacy commit-status API not having a
# ``cancelled`` enum value at all.
#
# The pattern matches both spellings (``cancelled`` and ``canceled``)
# at word boundaries, case-insensitively. ``cancellation`` and the
# like are deliberately NOT matched -- a description such as
# "Cancellation tests passed" must not trip this and reclassify a
# real failure as a cancellation.
_CANCELLED_DESCRIPTION_RE = re.compile(r"\bcancell?ed\b", re.IGNORECASE)

# Forgejo Actions also surfaces jobs that are gated on something the
# bot cannot fix (manual approval, required-environment review,
# branch-protection condition, ...) through the legacy
# commit-status API as ``status="pending"`` with
# ``description="Blocked by required conditions"``. These are
# never going to flip green on their own and they don't fail
# either: they sit indefinitely until a human acts in the UI. We
# treat them as a distinct logical state ``BLOCKED`` so the
# end-of-run summary can call them out for the operator alongside
# cancelled jobs. The description text is matched verbatim
# (case-insensitive) because no other Forgejo description starts
# with this exact phrase; loose matching on just ``\bblocked\b``
# would risk reclassifying unrelated "blocked on upstream issue"
# style descriptions.
_BLOCKED_DESCRIPTION_RE = re.compile(
    r"blocked by required conditions", re.IGNORECASE
)


def row_effective_state(row: ApiObject) -> str | None:
    """Canonical state of a single commit-status row.

    Wraps ``normalize_status_state`` on the row's state/status field
    and adds two description-based overrides for Forgejo Actions:

    - ``FAILURE`` / ``ERROR`` rows whose description matches
      ``_CANCELLED_DESCRIPTION_RE`` are reclassified as ``CANCELLED``.
    - ``PENDING`` rows whose description matches
      ``_BLOCKED_DESCRIPTION_RE`` are reclassified as ``BLOCKED``.

    This is the only place that knows about either Forgejo Actions
    shape; downstream helpers that need to ask "is this row
    cancelled / blocked / failing / pending" should call this rather
    than ``normalize_status_state`` directly.
    """
    raw_state = row.get("state") or row.get("status")
    state = normalize_status_state(raw_state)
    description = row.get("description")
    if not isinstance(description, str):
        return state
    if state in ("FAILURE", "ERROR") and _CANCELLED_DESCRIPTION_RE.search(description):
        return "CANCELLED"
    if state == "PENDING" and _BLOCKED_DESCRIPTION_RE.search(description):
        return "BLOCKED"
    return state


def effective_commit_statuses(statuses: list[ApiObject]) -> dict[str, tuple[str, datetime | None]]:
    ordered = sorted(
        statuses,
        key=lambda st: first_dt(st, "updated_at", "created_at")
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    result: dict[str, tuple[str, datetime | None]] = {}
    for status in ordered:
        context = status.get("context") or status.get("name") or status.get("target_url")
        if not isinstance(context, str) or not context:
            continue
        state = row_effective_state(status)
        if state is None:
            continue
        when = first_dt(status, "updated_at", "created_at")
        result[context] = (state, when)
    return result


# Commit contexts whose latest state is one of these get a possible
# ``ci_triage`` nag. PENDING / NEUTRAL etc. do not: we still skip those
# PRs early without calling the LLM (same as before, but without treating
# them as "failure" notifications).
CI_TRIAGE_NAG_STATES: frozenset[str] = frozenset({"FAILURE", "ERROR"})


def _status_row_time(st: ApiObject) -> datetime:
    return first_dt(st, "created_at", "updated_at") or datetime.min.replace(tzinfo=timezone.utc)


def group_commit_statuses_by_context(statuses: list[ApiObject]) -> dict[str, list[ApiObject]]:
    groups: dict[str, list[ApiObject]] = {}
    for st in statuses:
        if not isinstance(st, dict):
            continue
        context = st.get("context") or st.get("name") or st.get("target_url")
        if not isinstance(context, str) or not context:
            continue
        groups.setdefault(context, []).append(st)
    for rows in groups.values():
        rows.sort(key=_status_row_time)
    return groups


def extract_contexts_with_state(
    statuses: list[ApiObject], target_state: str,
) -> tuple[str, ...]:
    """Return the sorted, de-duplicated names of commit-status contexts
    whose *latest* effective state equals ``target_state``.

    Used by the end-of-run summary (and only by it) to surface PRs
    that need the operator to act in the Forgejo UI -- ``CANCELLED``
    contexts need a manual Rerun click, ``BLOCKED`` contexts need
    the gating condition to be released. Forgejo's HTTP API exposes
    neither a rerun endpoint nor a way to release required-condition
    gates (verified against the swagger spec; first observed on
    Forgejo 15.0.0+gitea-1.22.0), so fairy cannot drive these by
    itself. Both states are also excluded from the LLM CI-triage
    payload upstream (see ``CI_TRIAGE_NAG_STATES``), so the model is
    not asked to nag about jobs that just need a human.
    """
    return tuple(
        sorted(
            ctx
            for ctx, (state, _) in effective_commit_statuses(statuses).items()
            if state == target_state
        )
    )


def absolutize_target_url(url: str, base_url: str) -> str:
    """Resolve a Forgejo status ``target_url`` to a fully-qualified URL.

    Forgejo's ``GET /commits/<sha>/statuses`` endpoint returns
    ``target_url`` as a path-only string for some integrations (notably
    Forgejo Actions, which emits e.g. ``/<owner>/<repo>/actions/runs/<id>/jobs/<id>``).
    A bare path is not clickable in the LLM's reply text, so fairy was
    posting CI heads-up comments containing literal paths instead of
    real links. Resolve every ``target_url`` against the PR's
    ``html_url`` (which carries the scheme + host) so downstream
    consumers always see an absolute URL.

    ``urllib.parse.urljoin`` handles all three input shapes: leaves an
    already-absolute URL unchanged, replaces the path component for
    ``/path`` inputs, and resolves relative paths against the base.
    """
    if not url:
        return ""
    if not base_url:
        return url
    return urljoin(base_url, url)


def build_ci_failure_details(
    statuses: list[ApiObject],
    *,
    base_url: str = "",
) -> list[dict[str, object]]:
    """Return one object per context whose *latest* state is ERROR or FAILURE.

    ``base_url`` is the PR's ``html_url`` (or any URL on the same forge);
    it is used to absolutize each entry's ``target_url`` so the LLM
    receives clickable links rather than bare paths. When omitted the
    URLs are passed through verbatim (used by tests that do not need
    absolutization).
    """
    out: list[dict[str, object]] = []
    for context, rows in sorted(group_commit_statuses_by_context(statuses).items()):
        if not rows:
            continue
        last = rows[-1]
        last_state = row_effective_state(last)
        if last_state not in CI_TRIAGE_NAG_STATES:
            continue
        i = len(rows) - 1
        while i >= 0:
            st = row_effective_state(rows[i])
            if st not in CI_TRIAGE_NAG_STATES:
                break
            i -= 1
        streak = rows[i + 1 :]
        first_row = streak[0]
        first_at = first_dt(first_row, "created_at", "updated_at")
        last_at = first_dt(last, "created_at", "updated_at")
        desc = last.get("description")
        if not isinstance(desc, str):
            desc = ""
        target = last.get("target_url")
        if not isinstance(target, str):
            target = ""
        entry: dict[str, object] = {
            "context": context,
            "state": last_state or "",
            "description": desc,
            "target_url": absolutize_target_url(target, base_url),
        }
        if first_at is not None:
            entry["first_failing_at"] = first_at.isoformat()
        if last_at is not None:
            entry["last_failing_at"] = last_at.isoformat()
        out.append(entry)
    return out


def attach_ci_failure_logs(
    args: argparse.Namespace,
    details: list[dict[str, object]],
) -> None:
    """Enrich each failing context in ``details`` in place with ``log_tail``.

    Fetches up to ``args.ci_failure_log_lines`` trailing lines of each
    context's job log (see ``ci_log.fetch_job_log_tail``) so the triage and
    reviewer LLMs see the actual error rather than only the one-line status
    description. Call this only once a triager/reviewer is known to run, so
    no log is fetched for a PR that will be skipped. No-op when the flag is
    0; contexts whose log is unreachable are left unchanged.
    """
    max_lines = getattr(args, "ci_failure_log_lines", 0)
    if max_lines <= 0:
        return
    for d in details:
        url = d.get("target_url")
        if not isinstance(url, str) or not url:
            continue
        tail = ci_log.fetch_job_log_tail(url, max_lines=max_lines)
        if tail:
            d["log_tail"] = tail
            logger.debug(
                "PR CI log: attached %d-line tail for context=%r",
                len(tail.splitlines()), d.get("context"),
            )


def collect_self_comment_bodies(
    self_login: str | None,
    reviews: list[ApiObject],
    comments: list[ApiObject],
    review_comments: list[ApiObject],
) -> list[str]:
    if not self_login:
        return []
    bodies: list[str] = []
    for item in (*comments, *review_comments, *reviews):
        if get_item_author_login(item) != self_login:
            continue
        body = item.get("body")
        if isinstance(body, str) and body.strip():
            bodies.append(body)
    return bodies


# Forgejo/Gitea Actions append the workflow trigger event to a status
# context name, e.g. ``Test / Fate (Full, wine) (pull_request)`` or
# ``/ pr_labeler (pull_request_target)`` (captured from code.ffmpeg.org;
# see tests/test_cancelled_ci_handling.py fixtures). The event is always a
# snake_case token, so stripping a trailing ``(event)`` leaves a real
# matrix parenthetical -- ``(Full, wine)``, ``(32 bit)`` -- untouched.
_CI_EVENT_SUFFIX_RE = re.compile(r"\s*\([a-z_]+\)\s*$")


def strip_ci_event_suffix(context: str) -> str:
    return _CI_EVENT_SUFFIX_RE.sub("", context)


def fairy_mentioned_ci_context(
    context: str,
    target_url: str,
    bodies: list[str],
) -> bool:
    # Match on the event-suffix-stripped name: fairy quotes the bare job
    # name (``Test / Fate (Full, wine)``) in her heads-up, while the raw
    # context carries the ``(pull_request)`` suffix, so a verbatim
    # ``context in body`` never hit and she re-announced the same red jobs
    # after every push/CI re-run. target_url is per-run and cannot dedup
    # across re-runs, so the name is the durable signal.
    name = strip_ci_event_suffix(context)
    return any(
        (name and name in b) or (target_url and target_url in b)
        for b in bodies
    )


def partition_ci_announcement(
    details: list[dict[str, object]],
    bodies: list[str],
) -> tuple[list[str], list[str]]:
    """``mentioned`` and ``need`` are parallel lists of context names."""
    mentioned: list[str] = []
    need: list[str] = []
    for d in details:
        ctx = d.get("context")
        if not isinstance(ctx, str) or not ctx:
            continue
        url = d.get("target_url")
        url = url if isinstance(url, str) else ""
        if fairy_mentioned_ci_context(ctx, url, bodies):
            mentioned.append(ctx)
        else:
            need.append(ctx)
    return mentioned, need


def build_ci_triage_payload(
    head_sha: str,
    failure_details: list[dict[str, object]],
    fairy_bodies: list[str],
) -> JsonObject:
    men, need = partition_ci_announcement(failure_details, fairy_bodies)
    return {
        "head_sha": head_sha,
        "failure_contexts": failure_details,
        "contexts_bot_already_mentioned": men,
        "contexts_still_requiring_announcement": need,
    }


def get_pr_author(pr: ApiObject) -> str:
    user = pr.get("user") or {}
    for key in ("login", "username", "full_name"):
        value = user.get(key)
        if isinstance(value, str) and value:
            return value
    return "?"


def get_pr_head_ref(pr: ApiObject) -> str | None:
    head = pr.get("head")
    if isinstance(head, dict):
        for key in ("sha", "ref"):
            value = head.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("sha", "head_sha"):
        value = pr.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# Forgejo emits typed timeline comments for the "schedule auto-merge"
# and "cancel scheduled auto-merge" actions. Source:
# https://codeberg.org/forgejo/forgejo/src/branch/forgejo/models/issues/comment.go
#   * ``CommentTypePRScheduledToAutoMerge`` (= 34, string
#     ``"pull_scheduled_merge"``)
#   * ``CommentTypePRUnScheduledToAutoMerge`` (= 35, string
#     ``"pull_cancel_scheduled_merge"``)
# Behavior first observed on Forgejo 15.0.x; expected to be stable
# across releases since the constants are part of the public API
# enum and Gitea uses the same names.
_AUTO_MERGE_SCHEDULE_EVENT = "pull_scheduled_merge"
_AUTO_MERGE_CANCEL_EVENT = "pull_cancel_scheduled_merge"
_AUTO_MERGE_EVENTS = frozenset({_AUTO_MERGE_SCHEDULE_EVENT, _AUTO_MERGE_CANCEL_EVENT})


def auto_merge_state_from_timeline(timeline: list[ApiObject]) -> str:
    """Derive the current auto-merge schedule state from the PR timeline.

    Returns ``"merge"`` if the most recent of the two auto-merge events
    is a schedule, ``"no"`` if it is a cancellation OR no such events
    exist. Latest-event-wins, so a "scheduled, then canceled, then
    re-scheduled" history correctly resolves to ``"merge"``.

    The merge-vs-rebase distinction is intentionally collapsed.
    Forgejo stores the merge style on a separate ``pull_auto_merge``
    table that ``CreateAutoMergeComment`` does not copy onto the
    typed comment, and Forgejo's HTTP API has no GET endpoint that
    exposes the schedule (only ``DELETE /pulls/{n}/merge`` to cancel
    -- see ``routers/api/v1/api.go``). Distinguishing merge from
    rebase therefore has no API source; the value would only inform
    a tally line in the end-of-run summary, which is not worth a
    second round-trip per PR.
    """
    latest_ts: str | None = None
    latest_type: str | None = None
    for entry in timeline:
        ev_type = entry.get("type")
        if ev_type not in _AUTO_MERGE_EVENTS:
            continue
        ts = entry.get("created_at")
        if not isinstance(ts, str):
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_type = ev_type
    if latest_type == _AUTO_MERGE_SCHEDULE_EVENT:
        return "merge"
    return "no"


def get_auto_merge_info(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    timeline: list[ApiObject] | None = None,
) -> str:
    """Return the auto-merge schedule state for ``pr`` ("merge" / "no" / "?").

    Source: the PR's typed timeline (see ``forge_gcli.list_issue_timeline``
    and ``auto_merge_state_from_timeline``). On any fetch error this
    returns ``"?"`` so the caller can distinguish "definitely not
    scheduled" from "could not determine".

    ``timeline`` may be passed in by a caller that has already fetched
    the timeline for another purpose (e.g. push-event enrichment of the
    LLM discussion). When omitted the function fetches its own copy.
    A fetch failure inside this function still returns ``"?"`` because
    we cannot distinguish "scheduled" from "not scheduled" without the
    data; callers that pass in an explicitly-empty list have decided
    that absence-of-events means "no" and accept that semantics.
    """
    pr_number = pr.get("number")
    if not isinstance(pr_number, int):
        return "?"
    if timeline is None:
        try:
            timeline = list_issue_timeline(args, args.owner, args.repo, pr_number)
        except Exception as exc:
            logger.warning(
                "auto-merge: failed to fetch timeline for PR #%d: %s",
                pr_number, exc,
            )
            return "?"
    timeline = filter_activity_after(timeline, args.simulate_past, "created_at")
    return auto_merge_state_from_timeline(timeline)


def push_events_from_timeline(timeline: list[ApiObject]) -> list[DiscussionItem]:
    """Extract ``pull_push`` timeline entries as discussion-list items.

    Forgejo/Gitea emit one ``pull_push`` timeline event per push to the
    PR head branch. The interesting bit -- the new head SHA and the
    force-push flag -- lives in a JSON-string ``body`` field on the
    event, of the form
    ``{"is_force_push": bool, "commit_ids": ["<sha>", ...]}``. We
    surface those as synthetic ``kind="push"`` items so the triage and
    main reviewer prompts can see "after my last comment, the author
    pushed commit X" as a first-class signal, instead of having to
    infer it from a SHA mentioned in a prior bot comment vs the
    current ``head_sha`` (which the triager has been observed to
    speculate around -- see PR #23197 in the bot history).

    Malformed ``body`` strings are skipped silently; a push event we
    cannot decode is still better suppressed than turned into a hard
    failure that would block the whole review.
    """
    items: list[DiscussionItem] = []
    for entry in timeline:
        if entry.get("type") != "pull_push":
            continue
        raw_body = entry.get("body")
        if not isinstance(raw_body, str):
            continue
        try:
            decoded = json.loads(raw_body)
        except (ValueError, TypeError):
            continue
        if not isinstance(decoded, dict):
            continue
        commit_ids = decoded.get("commit_ids")
        if not isinstance(commit_ids, list):
            commit_ids = []
        commit_ids = [c for c in commit_ids if isinstance(c, str) and c]
        is_force_push = bool(decoded.get("is_force_push"))
        head_sha = commit_ids[-1] if commit_ids else None
        user = entry.get("user") or {}
        author = (
            user.get("login")
            or user.get("username")
            or user.get("full_name")
            or "?"
        )
        items.append({
            "kind": "push",
            "author": author,
            "created_at": entry.get("created_at"),
            "head_sha": head_sha,
            "is_force_push": is_force_push,
            "commit_count": len(commit_ids),
        })
    return items

class ReviewState(NamedTuple):
    state: str
    when: datetime | None
    # Forgejo/Gitea set ``stale`` when commits are pushed after the review
    # and ``dismissed`` on explicit dismissal; either voids an approval
    # (first observed on FFmpeg #20148: a force-push invalidated fairy's
    # approval). GitHub REST reviews carry neither field -- branch
    # protection rewrites the state to DISMISSED instead -- so there the
    # missing fields read False and behavior is unchanged.
    stale: bool


def effective_review_states(reviews: list[ApiObject]) -> dict[str, ReviewState]:
    ordered = sorted(
        reviews,
        key=lambda rv: first_dt(rv, "submitted_at", "updated_at", "created_at")
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    states: dict[str, ReviewState] = {}
    for review in ordered:
        user = review.get("user") or {}
        login = user.get("login") or user.get("username")
        if not isinstance(login, str) or not login:
            continue
        state = normalize_review_state(review.get("state"))
        if state is None:
            continue
        when = first_dt(review, "submitted_at", "updated_at", "created_at")
        stale = bool(review.get("stale")) or bool(review.get("dismissed"))
        states[login] = ReviewState(state, when, stale)
    return states


def has_review_by_user(reviews: list[ApiObject], login: str | None) -> bool:
    if not login:
        return False
    for review in reviews:
        user = review.get("user") or {}
        reviewer = user.get("login") or user.get("username")
        if isinstance(reviewer, str) and reviewer == login:
            return True
    return False


def effective_min_age_days(
    args: argparse.Namespace,
    reviews: list[ApiObject],
    self_login: str | None,
) -> float:
    d = float(args.min_age_days)
    if self_login and has_review_by_user(reviews, self_login):
        d = min(d, REVIEWED_PR_MIN_AGE_DAYS)
    return d


def compile_wip_regex(prefixes: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(p) for p in prefixes if p]
    if not escaped:
        escaped = [re.escape(p) for p in DEFAULT_WIP_PREFIXES]
    return re.compile(rf"^\s*(?:{'|'.join(escaped)})", re.IGNORECASE)


def is_marked_wip(pr: ApiObject, wip_re: re.Pattern[str]) -> bool:
    if bool(pr.get("draft")):
        return True
    title = pr.get("title")
    return isinstance(title, str) and bool(wip_re.search(title))


def get_item_author_login(item: ApiObject) -> str | None:
    user = item.get("user")
    if not isinstance(user, dict): #XXX check if needed must be on load not here
        return None
    login = user.get("login") or user.get("username") #XXX only one of the 3 is correct, if more can be corrrect then this code is totally wrong
    return login if isinstance(login, str) and login else None # again isinstance is wrong here


def compile_user_mention_regex(login: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_-])@{re.escape(login)}(?![A-Za-z0-9_-])", re.IGNORECASE)


def item_body_mentions_user(item: ApiObject, mention_re: re.Pattern[str]) -> bool:
    body = item.get("body")
    return isinstance(body, str) and bool(mention_re.search(body)) # wrong isinstance use


def filter_activity_after(
    items: list[ApiObject],
    cutoff: datetime | None,
    *time_keys: str,
) -> list[ApiObject]:
    if cutoff is None:
        return items
    kept: list[ApiObject] = []
    for item in items:
        when = first_dt(item, *time_keys)
        if when is None or when <= cutoff:
            kept.append(item)
    return kept


def get_last_activity(
    pr: ApiObject,
    reviews: list[ApiObject],
    comments: list[ApiObject],
    review_comments: list[ApiObject],
    predicate: ActivityPredicate | None = None,
    include_pr_updated: bool = True,
) -> datetime | None:
    latest = None
    if predicate is None:
        latest = first_dt(pr, "updated_at", "created_at") if include_pr_updated else first_dt(pr, "created_at")
        predicate = lambda i: True

    for item in reviews:
        if predicate(item):
            latest = max_dt([latest, first_dt(item, "submitted_at", "updated_at", "created_at")])

    for item in comments + review_comments:
        if predicate(item):
            latest = max_dt([latest, first_dt(item, "updated_at", "created_at")])

    return latest


def describe_age(now: datetime, dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    delta = now - dt
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def parse_label_changes(raw: object, allowlist: list[str]) -> tuple[LabelChange, ...]:
    """Parse the wrapper's ``label_changes`` into typed entries.

    This is the process boundary (subprocess stdout), so each entry is
    re-validated against ``allowlist`` here even though the wrapper
    already sanitized it. Bad labels, bad ``op``, and duplicate
    ``(label, op)`` pairs are dropped; with no allowlist nothing passes.
    """
    if not allowlist or not isinstance(raw, list):
        return ()
    allowed = frozenset(allowlist)
    out: list[LabelChange] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        op = item.get("op")
        if not isinstance(label, str) or label not in allowed or op not in ("add", "remove"):
            continue
        if (label, op) in seen:
            continue
        seen.add((label, op))
        reason = item.get("reason")
        out.append(LabelChange(
            label=label,
            op=op,
            reason=reason if isinstance(reason, str) else "",
            post=bool(item.get("post")),
        ))
    return tuple(out)


def label_names(changes: tuple[LabelChange, ...], op: str) -> tuple[str, ...]:
    return tuple(c.label for c in changes if c.op == op)


def decision_has_label_changes(decision: Decision) -> bool:
    return bool(decision.label_changes)


def manual_action_description(decision: Decision) -> str:
    parts: list[str] = []
    if decision.action in ACTIONABLE_DECISIONS:
        parts.append(decision.action.replace("_", "-"))
    if decision.label_changes:
        labels_part = (
            f"labels add={list(label_names(decision.label_changes, 'add'))!r} "
            f"remove={list(label_names(decision.label_changes, 'remove'))!r}"
        )
        reasons = "; ".join(
            f"{c.label}: {c.reason}" for c in decision.label_changes if c.reason
        )
        if reasons:
            labels_part += f" ({reasons})"
        parts.append(labels_part)
    return " + ".join(parts) if parts else decision.action


def apply_triage_labels(
    args: argparse.Namespace,
    prepared: PreparedItem,
    decision: Decision,
    *,
    skip_guard: bool,
) -> bool:
    """Apply label changes for ``decision``; False when the staleness
    guard suppressed them.

    When ``skip_guard`` is True the caller has just successfully run
    ``submit_decision_action``; re-checking ``check_pr_still_unchanged``
    would fail spuriously because Forgejo bumps ``pr.updated_at`` on
    the comment/approval we just posted.
    """
    if not skip_guard:
        changed_reason = check_pr_still_unchanged(args, prepared, decision)
        if changed_reason is not None:
            logger.info(
                "PR #%s: SKIP            label changes skipped because %s",
                decision.pr_number,
                changed_reason,
            )
            return False

    pr = get_pr(args, decision.pr_number)
    current = set(labels(pr))
    apply_issue_label_changes(
        args,
        args.owner,
        args.repo,
        decision.pr_number,
        list(label_names(decision.label_changes, "add")),
        list(label_names(decision.label_changes, "remove")),
        current,
    )
    post_label_explanations(args, decision.pr_number, decision.label_changes, current)
    return True


def post_label_explanations(
    args: argparse.Namespace,
    number: int,
    label_changes: tuple[LabelChange, ...],
    current_labels: set[str],
    *,
    kind: str = forge_gcli.KIND_PR,
) -> None:
    """Post the rationale for label changes the triager flagged ``post``.

    Only labels that ACTUALLY transition this run are explained: an add of
    a label not already present, or a remove of one that is. The PR's own
    label set is therefore the idempotency key -- a label explained once is
    not re-explained on later runs -- so no separate dedup state is needed.
    ``current_labels`` is the set captured before the apply above.
    """
    posted = [
        c for c in label_changes
        if c.post and c.reason
        and ((c.op == "add" and c.label not in current_labels)
             or (c.op == "remove" and c.label in current_labels))
    ]
    if not posted:
        return
    verb = {"add": "Added", "remove": "Removed"}
    lines = [f"- **{verb[c.op]} `{c.label}`**: {c.reason}" for c in posted]
    body = "Label changes:\n\n" + "\n".join(lines)
    logger.info(
        "%s #%s: posting label rationale for %d change(s): %s",
        kind, number,
        len(posted),
        ", ".join(f"{c.op} {c.label}" for c in posted),
    )
    post_issue_comment(args, args.owner, args.repo, number, body, kind=kind)


def apply_decision(
    args: argparse.Namespace,
    prepared: PreparedItem,
    decision: Decision,
    *,
    cache: gcli_cache.Cache | None,
    submitted_counts: dict[str, int],
) -> None:
    """Submit the action then apply labels.

    Labels run last so the action's PR-state guard is checked once on
    pristine ``updated_at``; a drifted action skips labels too because
    label apply would drift identically.
    """
    updated = workset_operator_review(args, "pr", decision)
    if updated is None:
        return
    decision = updated
    if decision.action in ACTIONABLE_DECISIONS:
        if not submit_decision_action(
            args, prepared, decision, cache=cache,
        ):
            return
        submitted_counts[decision.action] += 1
        if decision_has_label_changes(decision):
            apply_triage_labels(args, prepared, decision, skip_guard=True)
        workset_transition(args, "pr", decision.pr_number, workset.WorkState.POSTED)
        return
    if decision_has_label_changes(decision):
        if apply_triage_labels(args, prepared, decision, skip_guard=False):
            workset_transition(args, "pr", decision.pr_number, workset.WorkState.POSTED)


def prompt_manual(pr_number: int, action: str, pr_url: str = "") -> str:
    # Show the full PR URL when known so the operator can click /
    # paste it straight into a browser to inspect the PR before
    # answering. Falls back to the bare ``PR #N`` shape only when
    # the caller has no URL (e.g. early-exit decisions made before
    # any PR object was fetched).
    label = pr_url or f"PR #{pr_number}"
    while True:
        # Render the prompt on stderr -- the same stream the summary lines
        # and the live ``[wrapper ...]`` output use -- and read the answer
        # with a bare ``input()``. ``input()``'s own prompt would go to
        # stdout, which buffers independently of stderr, so it could flush
        # out of order relative to the surrounding log. The blank lines
        # framing the block are emitted by the caller, around the whole
        # summary+prompt block, not here.
        sys.stderr.write(
            f"Apply {action} for {label}? "
            f"{ANSI_BOLD_RED}[yes/skip/defer/quit/retry]{ANSI_RESET} "
        )
        sys.stderr.flush()
        try:
            answer = input().strip().lower()
        except EOFError:
            return "quit"

        if answer in ("y", "yes"):
            return "apply"
        if answer in ("", "s", "skip"):
            return "skip"
        if answer in ("d", "defer"):
            return "defer"
        if answer in ("q", "quit"):
            return "quit"
        if answer in ("r", "retry"):
            return "retry"

        logger.warning("Please answer yes, skip, defer, quit or retry.")


def would_auto_merge_after_approval(d: Decision) -> bool:
    return d.action == "approve" and d.auto_merge == "merge"


def format_action(d: Decision) -> str:
    action = d.action.upper().ljust(15)
    if would_auto_merge_after_approval(d):
        return f"{ANSI_RED}{action}{ANSI_RESET}"
    return action


def format_llm_classification(classification: str) -> str:
    mapping = {
        "-": "-",
        "approve": "ok",
        "minor_issues_approve": "minor",
        "moderate_issues": "moderate",
        "reply_no_verdict": "reply",
        "major_issues": "major",
        "skip": "skip",
        "error": "error",
        # pre-rename names persisted in bot state (last_llm_decision)
        "ok_approve": "ok",
        "moderate_issues_comment": "moderate",
        "major_request_changes": "major",
        "helpful_reply": "reply",
    }
    return mapping.get(classification, classification)



def build_llm_discussion(
    reviews: list[ApiObject],
    comments: list[ApiObject],
    review_comments: list[ApiObject],
    timeline: list[ApiObject] | None = None,
) -> list[DiscussionItem]:
    """Merge the PR's comments, reviews, review comments and (optionally)
    push events into a single chronologically-sorted list for the LLM.

    ``timeline`` is the raw ``/issues/{n}/timeline`` payload; only its
    ``pull_push`` entries are consumed (see ``push_events_from_timeline``).
    Passing ``None`` (or omitting it) yields a comments-only discussion.
    The push items carry ``kind="push"`` plus ``head_sha`` /
    ``is_force_push`` so the triage LLM can tell unambiguously that
    new code arrived after its prior comment instead of guessing from
    a SHA mentioned in the comment body.
    """
    items: list[DiscussionItem] = []
    if timeline:
        items.extend(push_events_from_timeline(timeline))

    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            body = ""
        urls = attachment_urls(comment)
        # a Forgejo comment can be an attachment with no text at all
        if not body.strip() and not urls:
            continue
        user = comment.get("user") or {}
        author = user.get("login") or user.get("username") or user.get("full_name") or "?"
        item: DiscussionItem = {
            "kind": "comment",
            "author": author,
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "body": body,
        }
        if urls:
            item["attachment_urls"] = urls
        items.append(item)

    for review in reviews:
        body = review.get("body")
        if not isinstance(body, str) or not body.strip():
            continue
        user = review.get("user") or {}
        author = user.get("login") or user.get("username") or user.get("full_name") or "?"
        items.append(
            {
                "kind": "review",
                "author": author,
                "state": review.get("state"),
                "submitted_at": review.get("submitted_at"),
                "updated_at": review.get("updated_at"),
                "body": body,
            }
        )

    for comment in review_comments:
        body = comment.get("body")
        if not isinstance(body, str) or not body.strip():
            continue
        user = comment.get("user") or {}
        author = user.get("login") or user.get("username") or user.get("full_name") or "?"
        items.append(
            {
                "kind": "review_comment",
                "author": author,
                "created_at": comment.get("created_at"),
                "updated_at": comment.get("updated_at"),
                "path": comment.get("path"),
                "line": comment.get("line") if comment.get("line") is not None else comment.get("position"),
                "side": comment.get("side"),
                "commit_id": comment.get("commit_id"),
                "body": body,
            }
        )

    items.sort(
        key=lambda item: iso_to_dt(
            item.get("submitted_at")
            or item.get("updated_at")
            or item.get("created_at")
        ) or datetime.min.replace(tzinfo=timezone.utc)
    )
    return items


def get_last_self_nonapproval_activity(
    reviews: list[ApiObject],
    comments: list[ApiObject],
    review_comments: list[ApiObject],
    self_login: str | None,
) -> datetime | None:
    if not self_login:
        return None

    times: list[datetime | None] = []

    for comment in comments:
        user = comment.get("user") or {}
        author = user.get("login") or user.get("username") or user.get("full_name")
        if author == self_login:
            times.append(first_dt(comment, "updated_at", "created_at"))

    for review in reviews:
        user = review.get("user") or {}
        author = user.get("login") or user.get("username") or user.get("full_name")
        if author != self_login:
            continue

        state = normalize_review_state(review.get("state"))
        if state in ("APPROVED", "REQUEST_REVIEW"):
            continue

        times.append(first_dt(review, "submitted_at", "updated_at", "created_at"))

    for comment in review_comments:
        user = comment.get("user") or {}
        author = user.get("login") or user.get("username") or user.get("full_name")
        if author == self_login:
            times.append(first_dt(comment, "updated_at", "created_at"))

    return max_dt(times)

def patch_shas_for_run(
    args: argparse.Namespace, pr: ApiObject,
) -> tuple[str, str]:
    """``(base, head)`` SHAs for the patch.

    Live: from the PR object. ``--simulate-past``: from refs the
    operator pinned in ``--patch-repo`` (typically a separate
    cutoff-prepped mirror). ``head`` resolves
    ``--patch-pr-ref-template.format(number=N)``; ``base`` is
    ``merge-base(head, pr.base.ref)`` so the diff is against the
    base branch's tip in that same prepped mirror.
    """
    if args.simulate_past is None:
        return pr.get("merge_base") or pr["base"]["sha"], pr["head"]["sha"]
    head = git_util.git_rev_parse(
        args.patch_repo, args.patch_pr_ref_template.format(number=pr["number"]),
    )
    return git_util.git_merge_base(args.patch_repo, head, pr["base"]["ref"]), head


def fetch_patch_for_llm(
    args: argparse.Namespace, base_sha: str, head_sha: str, max_bytes: int,
) -> tuple[str, bool]:
    data = git_util.git_format_patch_series(args.patch_repo, base_sha, head_sha)
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def flex_fallback_extra_args(cmd_str: str) -> list[str]:
    """Return wrapper args that downgrade ``flex`` -> ``default`` service tier.

    OpenAI's ``flex`` tier trades latency for cost: when the flex pool
    is over-subscribed, ``responses.create`` returns 429 / Cloudflare
    502 indefinitely and the wrapper's rate-limit retry loop spins
    until ``--llm-timeout`` fires. Burning a final retry on the same
    saturated flex queue is rarely productive; a single non-flex try
    typically completes in minutes, which is also what an operator
    would do by hand.

    The wrapper's ``--service-tier`` / ``--triage-service-tier`` are
    plain argparse ``store`` actions, so re-appending them is last-
    wins and overrides whatever the configured cmd specified. We only
    emit a fallback for tiers actually set to ``flex`` (in either
    ``--flag value`` or ``--flag=value`` form); other tiers and the
    "tier unset, use account default" case are left alone.
    """
    tokens = shlex.split(cmd_str)
    extra: list[str] = []
    for flag in ("--service-tier", "--triage-service-tier"):
        flex_set = any(
            (tok == flag and i + 1 < len(tokens) and tokens[i + 1] == "flex")
            or tok == f"{flag}=flex"
            for i, tok in enumerate(tokens)
        )
        if flex_set:
            extra += [flag, "default"]
    return extra


def podman_host_cmd_args(args: argparse.Namespace) -> list[str]:
    """Extra --llm-review-cmd flags that route LLM shell work into
    containers on the podman host(s), and the codex container host when a
    codex: model is used. Empty unless --podman-host / --codex-host is set."""
    cmd: list[str] = []
    hosts = getattr(args, "podman_host", None)
    if hosts:
        cmd += ["--podman", *[f"--shell-host={h}" for h in hosts]]
    codex_host = getattr(args, "codex_host", None)
    if codex_host:
        cmd += [f"--codex-host={codex_host}"]
    codex_home = getattr(args, "codex_home", None)
    if codex_home:
        cmd += [f"--codex-home={codex_home}"]
    return cmd


def invoke_llm_wrapper(
    args: argparse.Namespace,
    payload: JsonObject,
    *,
    number: int | None,
    allowed_classifications: frozenset[str],
    label_allowlist: list[str],
    extra_cmd_args: list[str] | None = None,
    stderr_tag: str = "pr",
) -> LLMReview:
    """Run --llm-review-cmd on ``payload`` and parse its verdict.

    ``allowed_classifications`` is the caller's verdict vocabulary
    (PR review vs issue analysis); anything else raises. ``stderr_tag``
    labels the live-streamed wrapper stderr lines.
    """
    cmd = shlex.split(args.llm_review_cmd)
    cmd += podman_host_cmd_args(args)
    if getattr(args, "simulate_past", None) is not None:
        cmd += [f"--simulate-past-cutoff={args.simulate_past.isoformat()}"]
    if number is not None:
        ws_path = workset_path(args, stderr_tag, number)
        if ws_path is not None:
            cmd += [f"--workset-file={ws_path}"]
    if extra_cmd_args:
        cmd += list(extra_cmd_args)
    stderr_prefix = f"[wrapper {stderr_tag}=#{number}] " if number is not None else "[wrapper] "
    cp = run_cmd(
        cmd,
        verbose=args.verbose,
        verbose_threshold=2,
        input_text=json.dumps(payload),
        timeout=args.llm_timeout,
        stderr_line_prefix=stderr_prefix,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"LLM review command failed with exit code {cp.returncode}; see stderr above"
        )

    data = load_json(cp.stdout, context="LLM review command")
    if not isinstance(data, dict):
        raise RuntimeError(f"LLM review command returned {type(data).__name__}, expected object")

    classification = data.get("classification")
    message = data.get("message", "")
    if not isinstance(classification, str):
        raise RuntimeError("LLM review JSON lacks string field 'classification'")
    if not isinstance(message, str):
        raise RuntimeError("LLM review JSON lacks string field 'message'")

    classification = classification.strip()
    if classification not in allowed_classifications:
        raise RuntimeError(f"unsupported LLM classification: {classification!r}")
    label_changes = parse_label_changes(data.get("label_changes"), label_allowlist)
    return LLMReview(
        classification=classification,
        message=message.strip(),
        label_changes=label_changes,
    )


def call_llm_with_retries(
    args: argparse.Namespace,
    number: int,
    invoke: Callable[[list[str] | None], LLMReview],
) -> LLMReview:
    """Apply the --llm-max-attempts retry policy around ``invoke``.

    ``invoke`` receives the attempt's extra wrapper args (the
    flex->default service-tier fallback on the final attempt, else
    None; see ``flex_fallback_extra_args``). Re-raises the last error
    when every attempt failed.
    """
    max_attempts = max(1, int(getattr(args, "llm_max_attempts", 1) or 1))
    retry_delay = max(0.0, float(getattr(args, "llm_retry_delay", 0.0) or 0.0))
    flex_fallback = flex_fallback_extra_args(args.llm_review_cmd or "")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        extra_cmd_args: list[str] | None = None
        if attempt == max_attempts and max_attempts > 1 and flex_fallback:
            extra_cmd_args = flex_fallback
            logger.warning(
                "LLM review #%s final attempt %d/%d: overriding flex tier with default (extra args: %s)",
                number, attempt, max_attempts, " ".join(flex_fallback),
            )
        try:
            review = invoke(extra_cmd_args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    "LLM review #%s attempt %d/%d failed: %s; retrying in %.1fs",
                    number, attempt, max_attempts, exc, retry_delay,
                )
                if retry_delay > 0:
                    time.sleep(retry_delay)
            else:
                logger.warning(
                    "LLM review #%s attempt %d/%d failed: %s; giving up",
                    number, attempt, max_attempts, exc,
                )
            continue
        if attempt > 1:
            logger.info(
                "LLM review #%s succeeded on attempt %d/%d",
                number, attempt, max_attempts,
            )
        return review
    assert last_exc is not None
    raise last_exc


def run_llm_review(
    args: argparse.Namespace,
    pr: ApiObject,
    auto_merge: str,
    discussion: list[DiscussionItem],
    reviewer_username: str | None,
    ci_triage: JsonObject | None = None,
    *,
    extra_cmd_args: list[str] | None = None,
    ignore_triage_skip: bool = False,
    force_engage: bool = False,
) -> LLMReview:
    if not args.llm_review_cmd:
        return LLMReview("approve", "")

    base_sha, head_sha = patch_shas_for_run(args, pr)
    patch_text, patch_truncated = fetch_patch_for_llm(
        args, base_sha, head_sha, args.llm_max_patch_bytes,
    )
    payload: JsonObject = {
        "pull_request": {
            "number": pr.get("number"),
            "title": pr.get("title") or "",
            "body": pr.get("body") or "",
            "author": get_pr_author(pr),
            "html_url": pr.get("html_url") or "",
            "base_ref": ((pr.get("base") or {}).get("ref") if isinstance(pr.get("base"), dict) else "") or "",
            "head_ref": ((pr.get("head") or {}).get("ref") if isinstance(pr.get("head"), dict) else "") or "",
            "head_sha": head_sha,
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "changed_files": pr.get("changed_files"),
            "auto_merge": auto_merge,
            "labels": labels(pr),
        },
        "patch": patch_text,
        "patch_truncated": patch_truncated,
        "discussion": discussion,
        "reviewer_username": reviewer_username or "",
    }
    label_allowlist: list[str] = args.triage_labels
    if label_allowlist:
        payload["triage_label_allowlist"] = label_allowlist
    if ci_triage is not None:
        payload["ci_triage"] = ci_triage
    if ignore_triage_skip:
        payload["ignore_triage_skip"] = True
    if force_engage:
        payload["force_engage"] = True

    number = pr.get("number")
    return invoke_llm_wrapper(
        args,
        payload,
        number=number if isinstance(number, int) else None,
        allowed_classifications=frozenset({
            "approve",
            "minor_issues_approve",
            "moderate_issues",
            "major_issues",
            "reply_no_verdict",
            "skip",
        }),
        label_allowlist=label_allowlist,
        extra_cmd_args=extra_cmd_args,
    )


def apply_llm_review(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    auto_merge: str,
    number: int,
    title: str,
    author: str,
    last_activity: datetime | None,
    base_reason: str,
    discussion: list[DiscussionItem],
    reviewer_username: str | None,
    ci_triage: JsonObject | None = None,
    ignore_triage_skip: bool = False,
    force_engage: bool = False,
) -> Decision:
    try:
        review = call_llm_with_retries(
            args,
            number,
            lambda extra_cmd_args: run_llm_review(
                args, pr, auto_merge, discussion, reviewer_username, ci_triage,
                extra_cmd_args=extra_cmd_args,
                ignore_triage_skip=ignore_triage_skip,
                force_engage=force_engage,
            ),
        )
    except Exception as exc:
        max_attempts = max(1, int(getattr(args, "llm_max_attempts", 1) or 1))
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "skip",
            f"LLM review failed after {max_attempts} attempt(s): {exc}",
            last_activity,
            "error",
            "",
        )

    return decision_from_review(
        review,
        number=number,
        title=title,
        author=author,
        auto_merge=auto_merge,
        last_activity=last_activity,
        base_reason=base_reason,
    )


def llm_skip_reason(message: str) -> str:
    """One-line reason for an LLM skip: the first line of the model's own
    explanation, never a narration of what skip means in general."""
    first = message.strip().splitlines()[0] if message.strip() else ""
    return f"LLM skip: {first}"[:160] if first else "LLM chose skip"


def decision_from_review(
    review: LLMReview,
    *,
    number: int,
    title: str,
    author: str,
    auto_merge: str,
    last_activity: datetime | None,
    base_reason: str,
) -> Decision:
    reason = base_reason
    label_kwargs = {"label_changes": review.label_changes}
    if review.classification == "approve":
        return Decision(
            number, title, author, auto_merge, "approve", reason, last_activity,
            review.classification, review.message, **label_kwargs,
        )
    if review.classification == "minor_issues_approve":
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "approve",
            f"{reason}; LLM: minor issues",
            last_activity,
            review.classification,
            review.message,
            **label_kwargs,
        )
    if review.classification == "moderate_issues":
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "comment",
            f"{reason}; LLM: moderate issues",
            last_activity,
            review.classification,
            review.message,
            **label_kwargs,
        )
    if review.classification == "reply_no_verdict":
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "comment",
            f"{reason}; LLM: helpful reply",
            last_activity,
            review.classification,
            review.message,
            **label_kwargs,
        )
    if review.classification == "major_issues":
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "request_changes",
            f"{reason}; LLM: major issues",
            last_activity,
            review.classification,
            review.message,
            **label_kwargs,
        )
    if review.classification == "skip":
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "skip",
            llm_skip_reason(review.message),
            last_activity,
            review.classification,
            review.message,
            **label_kwargs,
        )
    return Decision(number, title, author, auto_merge, "skip", "unexpected LLM result", last_activity, "error", "")


def apply_llm_review_to_prepared(
    args: argparse.Namespace,
    prepared: PreparedPR,
) -> Decision:
    return apply_llm_review(
        args,
        prepared.pr,
        auto_merge=prepared.auto_merge,
        number=prepared.number,
        title=prepared.title,
        author=prepared.author,
        last_activity=prepared.last_activity,
        base_reason=prepared.base_reason,
        discussion=prepared.discussion,
        reviewer_username=prepared.reviewer_username,
        ci_triage=prepared.ci_triage,
        ignore_triage_skip=prepared.ignore_triage_skip,
        force_engage=prepared.force_engage,
    )


# ---------------------------------------------------------------------------
# LLM skip-backoff cache: per-PR exponential backoff after an LLM "skip"
# ---------------------------------------------------------------------------
#
# When the LLM classifies a PR as ``"skip"`` it leaves no trace on the PR
# itself (no comment, no approval), so on the very next run nothing has
# changed, every gate passes again, and we burn another LLM call. This
# cache breaks that loop with strict exponential doubling: 24h after the
# 1st skip, 48h after the 2nd, 96h after the 3rd, ... If anything that
# the LLM cares about changes (a push or new discussion activity), the
# cache is bypassed; if the LLM ever decides non-skip, the counter resets.
# We never permanently cache a skip -- worst case fairy rechecks an
# "always skip" PR less and less often, but always eventually.
#
# "Change" is deliberately defined as ``head_sha + last_activity`` only
# (NOT raw ``pr.updated_at``): label / milestone / assignee tweaks bump
# ``pr.updated_at`` but neither of those, so they correctly do NOT
# invalidate the backoff. New comments / pushes / reviews always do.
# ---------------------------------------------------------------------------


def backoff_for_consecutive_skips(consec_skips: int) -> timedelta:
    """Strict doubling from 24h: 24h, 48h, 96h, 192h, ... (no policy cap).

    The exponent is structurally clamped at 25 only to avoid Python's
    ``timedelta`` overflow (``timedelta`` tops out at ~2.7M days; the
    clamp kicks in at 24 * 2**25 hours = ~92,000 years, i.e. never in
    practice). This is NOT a backoff cap -- it just keeps the math
    representable.
    """
    if consec_skips < 1:
        return timedelta(0)
    exponent = min(consec_skips - 1, 25)
    return timedelta(hours=24 * (2 ** exponent))


def compute_llm_skip_backoff(
    entry: dict[str, object],
    head_sha: str | None,
    last_activity: datetime | None,
    now: datetime,
) -> tuple[int, datetime] | None:
    """Decide whether the cached LLM skip suppresses another LLM call.

    Returns ``(consecutive_skip_count, next_eligible_at)`` if the gate
    should fire (caller should emit a free skip Decision), or ``None``
    otherwise. Pure function -- no side effects, no I/O.
    """
    if entry.get("last_llm_decision") != "skip":
        return None
    if entry.get("last_llm_head_sha") != head_sha:
        return None
    last_act_iso = last_activity.isoformat() if last_activity else None
    if entry.get("last_llm_last_activity_iso") != last_act_iso:
        return None
    last_at = iso_to_dt(entry.get("last_llm_at"))
    if last_at is None:
        return None
    consec_raw = entry.get("consecutive_skip_count", 0)
    n = consec_raw if isinstance(consec_raw, int) and consec_raw > 0 else 0
    if n < 1:
        return None
    eligible_at = last_at + backoff_for_consecutive_skips(n)
    if now >= eligible_at:
        return None
    return n, eligible_at


def prepare_pr(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    now: datetime,
    self_login: str | None,
    wip_re: re.Pattern[str],
    cache: gcli_cache.Cache,
    discussion_cache_max_age: timedelta,
) -> Decision | PreparedPR:
    number = int(pr["number"])
    title = str(pr.get("title") or "")
    author = get_pr_author(pr)
    pr_last_activity = first_dt(pr, "updated_at", "created_at")

    if number in args.force_skip_prs:
        return Decision(
            number,
            title,
            author,
            "-",
            "skip",
            "forced skip by --force-skip-pr",
            pr_last_activity,
        )

    # ``--force-review-pr`` means "review this PR no matter what".
    # The WIP/mergeable gates below are heuristics for the cron mode
    # -- when the operator named a PR explicitly we honor that and let
    # the LLM look at it. The closed/merged gate is the exception:
    # reviewing a non-open PR is opt-in via --force-review-non-open,
    # since the usual intent of forcing is a still-open PR.
    # ``--force-skip-pr`` still wins (handled above) per its
    # documented precedence.
    is_forced = number in args.force_review_prs

    if pr.get("state") != "open" and not (is_forced and args.force_review_non_open):
        return Decision(number, title, author, "-", "skip", "not open", pr_last_activity)

    entry = workset_backoff_entry(args, "pr", number)
    # Timeline fetches go through ``gcli_cache.get``: it serves a
    # cached copy when ``pr.updated_at`` matches and transparently
    # refetches when it advances, so both consumers (auto-merge
    # detection via ``auto_merge_state_from_timeline``, LLM-discussion
    # enrichment via ``push_events_from_timeline``) see the same single
    # live copy without an extra in-process memo.
    live_updated_at = iso_to_dt(pr.get("updated_at"))

    def get_timeline() -> list[ApiObject]:
        if live_updated_at is None:
            return []
        try:
            fields = gcli_cache.get(
                cache, args, "pulls", args.owner, args.repo, number, live_updated_at,
                "timeline", max_age=discussion_cache_max_age,
            )
        except Exception as exc:
            logger.warning(
                "LLM discussion enrichment: failed to fetch timeline for "
                "PR #%d: %s", number, exc,
            )
            return []
        return list(fields["timeline"])

    auto_merge: str | None = None

    def get_auto_merge() -> str:
        nonlocal auto_merge
        if auto_merge is None:
            auto_merge = get_auto_merge_info(args, pr, timeline=get_timeline())
        return auto_merge

    # Closure-bound: ``external_approvers`` is computed once in this
    # ``prepare_pr`` invocation (a few lines below, after reviews are
    # available) and read at skip() call time, so every skip Decision
    # automatically carries the current value without each call site
    # needing its own kwarg.
    external_approvers: tuple[str, ...] = ()

    def skip(
        reason: str,
        *,
        last_activity_value: datetime | None = pr_last_activity,
        cancelled_ci_contexts: tuple[str, ...] = (),
        blocked_ci_contexts: tuple[str, ...] = (),
        merge_ready: bool = False,
    ) -> Decision:
        # Skip paths do NOT fetch auto-merge state -- this used to be
        # served from a stale disk cache to give the log line some
        # signal, but the cache was right only "most of the time";
        # showing ``-`` (unknown) is honest and the skipped PR is by
        # definition not going through any auto-approve logic that
        # depends on the value.
        return Decision(
            number,
            title,
            author,
            "-",
            "skip",
            reason,
            last_activity_value,
            cancelled_ci_contexts=cancelled_ci_contexts,
            blocked_ci_contexts=blocked_ci_contexts,
            merge_ready=merge_ready,
            external_approvers=external_approvers,
        )

    if is_marked_wip(pr, wip_re) and not is_forced:
        return skip("marked WIP/draft")

    if not pr.get("mergeable") and not is_forced:
        return skip("has conflicts with the target branch")

    reviews, comments, review_comments = get_pr_discussion(
        args,
        pr,
        cache=cache,
        cache_max_age=discussion_cache_max_age,
    )
    # For CI nag deduplication, scan fairy's *full* comment history on this
    # PR (before --simulate-past filtering) so a prior heads-up is not re-sent.
    fairy_bodies_for_ci = collect_self_comment_bodies(
        self_login, reviews, comments, review_comments
    )
    ignore_after = args.simulate_past
    reviews = filter_activity_after(
        reviews,
        ignore_after,
        "submitted_at",
        "updated_at",
        "created_at",
    )
    comments = filter_activity_after(comments, ignore_after, "updated_at", "created_at")
    review_comments = filter_activity_after(
        review_comments,
        ignore_after,
        "updated_at",
        "created_at",
    )

    # Deliberately compute last_activity from discussion items only (reviews,
    # comments, review_comments) and NOT from ``pr.updated_at``. Forgejo bumps
    # ``pr.updated_at`` for label add/remove, assignee changes, milestone
    # edits, and other metadata changes that should not cause fairy to post
    # a second review reply to a PR it has already answered.
    last_activity = get_last_activity(
        pr,
        reviews,
        comments,
        review_comments,
        include_pr_updated=False,
    )

    # A bare force-push (new head, no comment) IS real activity, but
    # ``pr.updated_at`` is too noisy to use as the signal (see above). Take
    # the push time straight from the timeline's ``pull_push`` events so a
    # re-pushed PR is reconsidered instead of being frozen at fairy's last
    # comment by the "no activity since prior non-approval" gate below
    # (FFmpeg #22961 and the wider haasn cohort sat skipped after silent
    # force-pushes). ``get_timeline`` is the same single live copy the
    # auto-merge / LLM-discussion consumers already read.
    latest_push = max_dt([
        first_dt(item, "created_at")
        for item in push_events_from_timeline(
            filter_activity_after(get_timeline(), ignore_after, "created_at")
        )
    ])
    if latest_push is not None and (last_activity is None or latest_push > last_activity):
        logger.debug(
            "PR #%d: latest push %s newer than discussion activity %s; "
            "treating the push as last activity",
            number,
            latest_push.isoformat(),
            last_activity.isoformat() if last_activity else "-",
        )
        last_activity = latest_push

    # Compute review state once, here, so the value is available to the
    # end-of-run "external approvers" reminder list even on PRs that
    # are about to early-skip via the backoff gate just below or the
    # forced-review path further down. The same ``states`` value is
    # re-used by the blockers / self-approved gates further down (no
    # second pass).
    states = effective_review_states(reviews)
    has_blocker = any(
        rs.state == "CHANGES_REQUESTED" for rs in states.values()
    )
    if not has_blocker:
        # Stale approvals are excluded: the forge no longer counts them
        # towards mergeability, so listing the PR as "approved and
        # waiting to be merged" would tell the operator to merge
        # something the forge will refuse.
        external_approvers = tuple(sorted(
            login for login, rs in states.items()
            if rs.state == "APPROVED" and not rs.stale and login != self_login
        ))
        # An auto-merge-queued PR will merge itself when CI flips green
        # (or already has); listing it as "needs human merge" is noise.
        # Pay one ``get_auto_merge`` call only for PRs that would
        # otherwise show up in the reminder.
        if external_approvers and get_auto_merge() == "merge":
            external_approvers = ()

    # LLM skip-backoff gate: if the LLM previously decided "skip" on
    # this exact (head_sha, last_activity) state and the doubling
    # window has not elapsed yet, return a free skip Decision now.
    # New comments / pushes bypass the gate (they bump head_sha or
    # last_activity); ``--force-review-pr`` bypasses it too, matching
    # the state/WIP/mergeable gates above -- an explicitly named PR is
    # reviewed no matter what.
    backoff_info = (
        None if is_forced
        else compute_llm_skip_backoff(entry, get_pr_head_ref(pr), last_activity, now)
    )
    if backoff_info is not None:
        consec, eligible_at = backoff_info
        backoff_window = backoff_for_consecutive_skips(consec)
        return skip(
            f"in LLM skip-backoff window after {consec} consecutive skip(s); "
            f"window={backoff_window}; next eligible at {eligible_at.isoformat()}",
            last_activity_value=last_activity,
        )

    forced_review_reason: str | None = None
    if number in args.force_review_prs:
        forced_review_reason = "forced review by --force-review-pr"
    elif self_login:
        # First collect every signal that says "fairy ought to reply
        # on this PR". Only if at least one such signal exists do we
        # bother computing whether fairy already replied to it.

        # Signal 1: someone @-mentioned self_login.
        mention_re = compile_user_mention_regex(self_login)
        latest_mention = get_last_activity(
            pr,
            reviews,
            comments,
            review_comments,
            predicate=lambda item: get_item_author_login(item) != self_login and item_body_mentions_user(item, mention_re),
        )

        # Signal 2: self_login was added as a requested reviewer (Forgejo
        # writes a REQUEST_REVIEW review entry whose user is the
        # requested reviewer when someone assigns us).
        latest_review_request = get_last_activity(
            pr,
            reviews,
            [],
            [],
            predicate=lambda item: (
                get_item_author_login(item) == self_login
                and normalize_review_state(item.get("state")) == "REQUEST_REVIEW"
            ),
        )

        if latest_mention is not None or latest_review_request is not None:
            # Last "real" activity by fairy on this PR. Excludes the
            # REQUEST_REVIEW pseudo-entry, which otherwise would always
            # look like fairy has already acted on the PR.
            reviewer_last_activity = get_last_activity(
                pr,
                reviews,
                comments,
                review_comments,
                predicate=lambda item: (
                    get_item_author_login(item) == self_login
                    and normalize_review_state(item.get("state")) != "REQUEST_REVIEW"
                ),
            )

            if latest_mention is not None and (
                reviewer_last_activity is None or latest_mention > reviewer_last_activity
            ):
                forced_review_reason = f"later discussion mentions reviewer {self_login}"
            elif latest_review_request is not None and (
                reviewer_last_activity is None or latest_review_request > reviewer_last_activity
            ):
                forced_review_reason = f"review requested from {self_login}"
                logger.debug(
                    "PR #%d forced review via REQUEST_REVIEW: request_at=%s reviewer_last_activity=%s",
                    number,
                    latest_review_request.isoformat(),
                    reviewer_last_activity.isoformat() if reviewer_last_activity else "-",
                )

    if forced_review_reason is not None:
        auto_merge_value = get_auto_merge()
        base_reason = forced_review_reason
        if not args.llm_review_cmd:
            return skip(
                f"{base_reason}; --llm-review-cmd not set (no automatic reply or approval)",
                last_activity_value=last_activity,
            )
        # Always give the LLM the CI context when CI is red, even on
        # forced review. A human asking "please look at this PR" while CI
        # is failing should still be answered in light of those failures,
        # not in a vacuum.
        head_ref_for_ci = get_pr_head_ref(pr)
        ci_triage_payload: JsonObject | None = None
        cancelled_ctxs: tuple[str, ...] = ()
        blocked_ctxs: tuple[str, ...] = ()
        if head_ref_for_ci:
            raw_for_ci = list_commit_statuses(args, head_ref_for_ci)
            fd = build_ci_failure_details(
                raw_for_ci, base_url=pr.get("html_url") or ""
            )
            cancelled_ctxs = extract_contexts_with_state(raw_for_ci, "CANCELLED")
            blocked_ctxs = extract_contexts_with_state(raw_for_ci, "BLOCKED")
            if cancelled_ctxs or blocked_ctxs:
                logger.debug(
                    "PR #%d: %d cancelled, %d blocked CI context(s) recorded "
                    "for end-of-run manual-action summary; cancelled=%s "
                    "blocked=%s",
                    number,
                    len(cancelled_ctxs),
                    len(blocked_ctxs),
                    ", ".join(cancelled_ctxs) or "-",
                    ", ".join(blocked_ctxs) or "-",
                )
            if fd:
                attach_ci_failure_logs(args, fd)
                ci_triage_payload = build_ci_triage_payload(
                    head_ref_for_ci, fd, fairy_bodies_for_ci
                )
                if args.verbose:
                    logger.debug(
                        "PR #%d: forced review; attached ci_triage with %d "
                        "ERROR/FAILURE context(s)",
                        number,
                        len(fd),
                    )
        return PreparedPR(
            pr=pr,
            number=number,
            title=title,
            author=author,
            auto_merge=auto_merge_value,
            last_activity=last_activity,
            base_reason=base_reason,
            discussion=build_llm_discussion(reviews, comments, review_comments, get_timeline()),
            reviewer_username=self_login,
            ci_triage=ci_triage_payload,
            cancelled_ci_contexts=cancelled_ctxs,
            blocked_ci_contexts=blocked_ctxs,
            external_approvers=external_approvers,
            ignore_triage_skip=is_forced and args.force_review_skip,
            force_engage=is_forced and args.force_engage,
        )

    # ``states`` and ``has_blocker`` were computed earlier (right after
    # ``last_activity``) so the end-of-run external-approvers summary
    # could see backoff'd / forced-review PRs. Re-use the same value
    # here.
    if has_blocker:
        blockers = sorted(
            login for login, rs in states.items()
            if rs.state == "CHANGES_REQUESTED"
        )
        return skip(f"outstanding change requests from: {', '.join(blockers)}", last_activity_value=last_activity)

    if self_login and not args.include_self_approved:
        self_state = states.get(self_login)
        if self_state and self_state.state == "APPROVED" and not self_state.stale:
            # Bot has already approved this PR. If auto-merge is not
            # scheduled, surface it in the end-of-run "needs merging"
            # reminder so an operator can hit the merge button. We do
            # not re-verify CI is still green -- fairy only approves
            # green CI, and any subsequent CI flip is already surfaced
            # by the cancelled/blocked summary on its own next run.
            #
            # A stale approval (code pushed after it; the forge no longer
            # counts it) deliberately falls through so the PR is
            # reconsidered for review (regression: FFmpeg #20148 sat
            # skipped forever after force-pushes voided the approval).
            return skip(
                f"already approved by {self_login}",
                last_activity_value=last_activity,
                merge_ready=get_auto_merge() != "merge",
            )

    if last_activity is None:
        return skip("cannot determine activity timestamp", last_activity_value=None)

    min_age_days = effective_min_age_days(args, reviews, self_login)
    if last_activity > now - timedelta(days=min_age_days):
        return skip(
            "activity is newer than threshold",
            last_activity_value=last_activity,
        )

    # Fetch CI status BEFORE the "no activity since prior non-approval"
    # gate so that gate's skip can carry cancelled/blocked context info
    # to the end-of-run manual-action summary. The previous layout fetched
    # CI only AFTER the gate, which meant long-stalled PRs (the cohort
    # most likely to have CI gated on a manual rerun / required-condition
    # release) silently fell off the summary.
    #
    # Cohort: this runs only after WIP/conflict/CHANGES_REQUESTED/
    # self-approved/last_activity-None/threshold-fresh have all skipped,
    # so the extra API call is bounded to PRs fairy would have looked
    # at anyway (just a bit later). Not cached: we have not verified
    # that ``pr.updated_at`` reliably bumps on commit-status row changes
    # in Forgejo, so the cache's (pr.updated_at + TTL) pattern cannot
    # be mirrored here without risking stale CI surface -- and per
    # CONTRIBUTING.md a cache that can lie is worse than no cache. A
    # pure-TTL cache is also off the table for the same reason.
    head_ref = get_pr_head_ref(pr)
    if not head_ref:
        return skip("cannot determine PR head commit", last_activity_value=last_activity)

    raw_status_list = list_commit_statuses(args, head_ref)
    # ``CANCELLED`` and ``BLOCKED`` are surfaced separately for the
    # operator-facing summary at end-of-run: Forgejo's HTTP API offers
    # no rerun endpoint and no way to release blocked-by-required-
    # conditions gates, so a human has to click. Both states are also
    # hidden from the LLM CI-triage payload below (see
    # ``CI_TRIAGE_NAG_STATES``) so the model is not prompted to nag
    # about something that just needs the UI Rerun button.
    cancelled_ctxs = extract_contexts_with_state(raw_status_list, "CANCELLED")
    blocked_ctxs = extract_contexts_with_state(raw_status_list, "BLOCKED")
    if cancelled_ctxs or blocked_ctxs:
        logger.debug(
            "PR #%d: %d cancelled, %d blocked CI context(s) recorded for "
            "end-of-run manual-action summary; cancelled=%s blocked=%s",
            number,
            len(cancelled_ctxs),
            len(blocked_ctxs),
            ", ".join(cancelled_ctxs) or "-",
            ", ".join(blocked_ctxs) or "-",
        )

    last_self_nonapproval = get_last_self_nonapproval_activity(reviews, comments, review_comments, self_login)
    if last_self_nonapproval is not None and last_activity <= last_self_nonapproval:
        return skip(
            f"no activity since prior non-approval message by {self_login}",
            last_activity_value=last_activity,
            cancelled_ci_contexts=cancelled_ctxs,
            blocked_ci_contexts=blocked_ctxs,
        )

    commit_statuses = effective_commit_statuses(raw_status_list)
    if not commit_statuses:
        return skip(
            "no commit statuses / CI results found",
            last_activity_value=last_activity,
            cancelled_ci_contexts=cancelled_ctxs,
            blocked_ci_contexts=blocked_ctxs,
        )

    failing_contexts = sorted(
        ctx for ctx, (state, _) in commit_statuses.items() if state != "SUCCESS"
    )
    # ``CANCELLED`` contexts are still in ``failing_contexts`` (they are
    # not ``SUCCESS``), so the PR is correctly NOT auto-approved when only
    # cancelled jobs exist.
    if failing_contexts:
        preview = ", ".join(failing_contexts[:4])
        if len(failing_contexts) > 4:
            preview += f", +{len(failing_contexts) - 4} more"
        failure_details = build_ci_failure_details(
            raw_status_list, base_url=pr.get("html_url") or ""
        )
        if not failure_details:
            # Only pending / neutral / cancelled / non-error states — same
            # early exit as before, without invoking the LLM. When the only
            # non-success contexts are ``CANCELLED``, ``cancelled_ctxs`` is
            # non-empty and the end-of-run summary will surface this PR.
            return skip(
                f"CI not successful: {preview}",
                last_activity_value=last_activity,
                cancelled_ci_contexts=cancelled_ctxs,
                blocked_ci_contexts=blocked_ctxs,
            )
        if args.triage_on_ci_failure:
            if not args.llm_review_cmd:
                return skip(
                    "CI not successful (--triage-on-ci-failure requires --llm-review-cmd); "
                    f"{preview}",
                    last_activity_value=last_activity,
                    cancelled_ci_contexts=cancelled_ctxs,
                    blocked_ci_contexts=blocked_ctxs,
                )
            if "--triage-model" not in (args.llm_review_cmd or ""):
                return skip(
                    "CI not successful; --triage-on-ci-failure needs --triage-model in --llm-review-cmd; "
                    f"{preview}",
                    last_activity_value=last_activity,
                    cancelled_ci_contexts=cancelled_ctxs,
                    blocked_ci_contexts=blocked_ctxs,
                )
            ci_payload = build_ci_triage_payload(
                get_pr_head_ref(pr) or "", failure_details, fairy_bodies_for_ci
            )
            men = ci_payload["contexts_bot_already_mentioned"]
            need = ci_payload["contexts_still_requiring_announcement"]
            assert isinstance(men, list) and isinstance(need, list)
            if not need:
                if args.verbose:
                    logger.info(
                        "PR #%d: skipping LLM (CI triage); bot already mentioned all %d "
                        "failing job context(s): %s",
                        number,
                        len(failure_details),
                        ", ".join(men) if men else "-",
                    )
                return skip(
                    "CI not successful; bot already mentioned all current "
                    f"ERROR/FAILURE job(s): {', '.join(men)}"
                    if men
                    else "CI not successful; bot already mentioned all current ERROR/FAILURE job(s)",
                    last_activity_value=last_activity,
                    cancelled_ci_contexts=cancelled_ctxs,
                    blocked_ci_contexts=blocked_ctxs,
                )
            # CI-failure triage is a proactive LLM action, so honor the
            # same discussion-inactivity gate as the normal approval path.
            # Forced-review signals (@mention / requested reviewer) have
            # already been handled earlier and never reach here.
            triage_min = effective_min_age_days(args, reviews, self_login)
            if last_activity > now - timedelta(days=triage_min):
                return skip(
                    "CI ERROR/FAILURE: last activity is newer than --min-age-days "
                    f"({triage_min}); not running CI triage yet",
                    last_activity_value=last_activity,
                    cancelled_ci_contexts=cancelled_ctxs,
                    blocked_ci_contexts=blocked_ctxs,
                )
            if args.verbose:
                logger.info(
                    "PR #%d: CI triage: %d job(s) still need announcement: %s",
                    number,
                    len(need),
                    ", ".join(need),
                )
            attach_ci_failure_logs(args, failure_details)
            auto_merge_value = get_auto_merge()
            base_reason = (
                f"CI triage (head not green): {len(need)} job(s) not yet "
                f"mentioned by bot: {', '.join(need[:6])}"
                f"{'...' if len(need) > 6 else ''}"
            )
            return PreparedPR(
                pr=pr,
                number=number,
                title=title,
                author=author,
                auto_merge=auto_merge_value,
                last_activity=last_activity,
                base_reason=base_reason,
                discussion=build_llm_discussion(reviews, comments, review_comments, get_timeline()),
                reviewer_username=self_login,
                ci_triage=ci_payload,
                cancelled_ci_contexts=cancelled_ctxs,
                blocked_ci_contexts=blocked_ctxs,
                external_approvers=external_approvers,
            )
        return skip(
            f"CI not successful: {preview}",
            last_activity_value=last_activity,
            cancelled_ci_contexts=cancelled_ctxs,
            blocked_ci_contexts=blocked_ctxs,
        )

    auto_merge_value = get_auto_merge()
    base_reason = f"matches all rules; CI contexts={len(commit_statuses)}"
    if args.llm_review_cmd:
        return PreparedPR(
            pr=pr,
            number=number,
            title=title,
            author=author,
            auto_merge=auto_merge_value,
            last_activity=last_activity,
            base_reason=base_reason,
            discussion=build_llm_discussion(reviews, comments, review_comments, get_timeline()),
            reviewer_username=self_login,
            external_approvers=external_approvers,
        )

    return Decision(
        number,
        title,
        author,
        auto_merge_value,
        "approve",
        base_reason,
        last_activity,
        expected_pr_updated_at=pr.get("updated_at"),
        expected_head_ref=get_pr_head_ref(pr),
    )


def safe_prepare_pr(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    now: datetime,
    self_login: str | None,
    wip_re: re.Pattern[str],
    cache: gcli_cache.Cache,
    discussion_cache_max_age: timedelta,
) -> PreparedItem:
    try:
        return prepare_pr(
            args,
            pr,
            now=now,
            self_login=self_login,
            wip_re=wip_re,
            cache=cache,
            discussion_cache_max_age=discussion_cache_max_age,
        )
    except Exception as exc:
        number = pr.get("number", "?")
        title = str(pr.get("title") or "")
        return Decision(
            int(number) if str(number).isdigit() else -1,
            title,
            get_pr_author(pr),
            "-",
            "error",
            str(exc),
            None,
            "error",
            "",
        )


def safe_apply_llm_review_to_prepared(
    args: argparse.Namespace,
    prepared: PreparedPR,
) -> Decision:
    # ``cancelled_ci_contexts`` / ``blocked_ci_contexts`` were computed
    # during ``prepare_pr`` from the raw status list (which is not threaded
    # into the LLM-review path), so we propagate them from the
    # ``PreparedPR`` into the resulting ``Decision`` here.
    # ``apply_llm_review_to_prepared`` constructs many different
    # ``Decision`` shapes via ``apply_llm_review``; rather than plumbing
    # the fields through every one of them, we patch them on at the single
    # funnel point. ``dataclasses.replace`` keeps the rest of the decision
    # intact.
    try:
        cached = workset_reusable_review(
            args, "pr", prepared.number,
            expected_updated_at=prepared.pr.get("updated_at"),
            expected_head_ref=get_pr_head_ref(prepared.pr),
            forced=prepared.number in args.force_review_prs,
        )
        decision = (
            decision_from_review(
                cached,
                number=prepared.number,
                title=prepared.title,
                author=prepared.author,
                auto_merge=prepared.auto_merge,
                last_activity=prepared.last_activity,
                base_reason=prepared.base_reason,
            )
            if cached is not None
            else apply_llm_review_to_prepared(args, prepared)
        )
    except Exception as exc:
        return Decision(
            prepared.number,
            prepared.title,
            prepared.author,
            prepared.auto_merge,
            "error",
            str(exc),
            prepared.last_activity,
            "error",
            "",
            cancelled_ci_contexts=prepared.cancelled_ci_contexts,
            blocked_ci_contexts=prepared.blocked_ci_contexts,
            external_approvers=prepared.external_approvers,
        )
    if prepared.cancelled_ci_contexts and not decision.cancelled_ci_contexts:
        decision = dataclasses_replace(
            decision, cancelled_ci_contexts=prepared.cancelled_ci_contexts
        )
    if prepared.blocked_ci_contexts and not decision.blocked_ci_contexts:
        decision = dataclasses_replace(
            decision, blocked_ci_contexts=prepared.blocked_ci_contexts
        )
    if prepared.external_approvers and not decision.external_approvers:
        decision = dataclasses_replace(
            decision, external_approvers=prepared.external_approvers
        )
    return decision


def workset_path(args: argparse.Namespace, kind: str, number: int) -> Path | None:
    """Item file for this run's repo; ``kind`` is "pr" | "issue". None for
    bare test namespaces without --workset-dir (all writes then no-op)."""
    root = getattr(args, "workset_dir", None)
    if not root:
        return None
    return workset.item_path(
        Path(root),
        forge_type=args.forge_type,
        account=args.gcli_account or "",
        owner=args.owner,
        repo=args.repo,
        kind=kind,
        number=number,
    )


def workset_repo_dir(args: argparse.Namespace) -> Path | None:
    """This run's per-repo workset directory; None without --workset-dir."""
    root = getattr(args, "workset_dir", None)
    if not root:
        return None
    return workset.repo_dir(
        Path(root),
        forge_type=args.forge_type,
        account=args.gcli_account or "",
        owner=args.owner,
        repo=args.repo,
    )


def workset_record_queued(
    args: argparse.Namespace, kind: str, *, number: int, title: str, html_url: str,
) -> None:
    path = workset_path(args, kind, number)
    if path is None:
        return
    existing = workset.load_item(path)
    if existing is not None:
        logger.debug("workset: keeping %s state=%s", path, existing.state.name)
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    workset.save_item(path, workset.WorkItem(
        kind=kind,
        forge_type=args.forge_type,
        account=args.gcli_account or "",
        owner=args.owner,
        repo=args.repo,
        number=number,
        state=workset.WorkState.QUEUED,
        created_at=now_iso,
        state_changed_at=now_iso,
        title=title,
        html_url=html_url,
    ))


def workset_record_reviewed(
    args: argparse.Namespace,
    kind: str,
    decision: Decision,
    *,
    expected_updated_at: str | None,
    expected_head_ref: str | None,
) -> None:
    """Persist the LLM verdict; an operator-deleted file stays deleted."""
    path = workset_path(args, kind, decision.pr_number)
    if path is None:
        return

    def record(item: workset.WorkItem) -> None:
        now = datetime.now(timezone.utc)
        if decision.llm_classification == "error":
            item.error = decision.reason
            item.set_state(workset.WorkState.ERROR, now)
            return
        item.error = None
        item.review = workset.ReviewResult(
            classification=decision.llm_classification,
            message=decision.llm_message,
            label_changes=[
                workset.LabelChange(label=c.label, op=c.op, reason=c.reason, post=c.post)
                for c in decision.label_changes
            ],
        )
        item.expected_updated_at = expected_updated_at
        item.expected_head_ref = expected_head_ref
        item.last_activity_iso = (
            decision.last_activity.isoformat() if decision.last_activity else None
        )
        item.llm_at = now.isoformat()
        if decision.llm_classification == "skip":
            item.consecutive_skip_count += 1
        else:
            item.consecutive_skip_count = 0
        item.set_state(workset.WorkState.REVIEWED, now)

    if workset.update_item(path, record) is None:
        logger.warning(
            "workset: %s #%d reviewed but %s is gone; verdict not persisted",
            kind, decision.pr_number, path,
        )


def workset_backoff_entry(args: argparse.Namespace, kind: str, number: int) -> dict[str, object]:
    """Skip-backoff view of the item file, in ``compute_llm_skip_backoff``'s
    entry vocabulary. Empty when no persisted verdict exists."""
    path = workset_path(args, kind, number)
    item = workset.load_item(path) if path is not None else None
    if item is None or item.state is not workset.WorkState.REVIEWED or item.review is None:
        return {}
    return {
        "last_llm_decision": item.review.classification,
        "last_llm_at": item.llm_at,
        "last_llm_head_sha": item.expected_head_ref,
        "last_llm_last_activity_iso": item.last_activity_iso,
        "consecutive_skip_count": item.consecutive_skip_count,
    }


def workset_reusable_review(
    args: argparse.Namespace,
    kind: str,
    number: int,
    *,
    expected_updated_at: str | None,
    expected_head_ref: str | None,
    forced: bool = False,
) -> LLMReview | None:
    """Persisted verdict for an unchanged item, or None to run the LLM.

    ``forced`` (--force-review-*) always re-runs, like every other gate.
    A persisted "skip" is never reused: that would make the exponential
    skip-backoff (see compute_llm_skip_backoff) permanent.
    """
    if forced:
        return None
    path = workset_path(args, kind, number)
    if path is None:
        return None
    item = workset.load_item(path)
    if item is None or item.state is not workset.WorkState.REVIEWED or item.review is None:
        return None
    if item.review.classification in ("skip", "error", "-", ""):
        logger.debug(
            "workset: %s #%d persisted %r is not reusable",
            kind, number, item.review.classification,
        )
        return None
    if (item.expected_updated_at != expected_updated_at
            or item.expected_head_ref != expected_head_ref):
        logger.info(
            "%s #%d: persisted review is stale (updated_at %r -> %r, head %r -> %r);"
            " re-reviewing",
            kind, number, item.expected_updated_at, expected_updated_at,
            item.expected_head_ref, expected_head_ref,
        )
        return None
    logger.info("%s #%d: reusing persisted review from %s (guard match)", kind, number, path)
    return workset_llm_review(item)


def workset_llm_review(item: workset.WorkItem) -> LLMReview:
    """The item file's persisted verdict as an ``LLMReview``. The caller
    has checked ``item.review`` is present."""
    assert item.review is not None
    return LLMReview(
        item.review.classification,
        item.review.message,
        tuple(
            LabelChange(c.label, c.op, c.reason, c.post)
            for c in item.review.label_changes
        ),
    )


def workset_table_decision(args: argparse.Namespace, number: int) -> Decision | None:
    """Rebuild a postable PR Decision purely from the item file, for
    operator table actions (fairy-ui) that hold no in-memory prepared
    object -- e.g. reviews left by a previous run. The file's guard rides
    on the Decision, so ``check_pr_still_unchanged`` still pins the post
    to the reviewed state."""
    path = workset_path(args, "pr", number)
    item = workset.load_item(path) if path is not None else None
    if item is None or item.state is not workset.WorkState.REVIEWED or item.review is None:
        return None
    decision = decision_from_review(
        workset_llm_review(item),
        number=number,
        title=item.title,
        author="",
        auto_merge="-",
        last_activity=iso_to_dt(item.last_activity_iso),
        base_reason="persisted review",
    )
    return dataclasses_replace(
        decision,
        expected_pr_updated_at=item.expected_updated_at,
        expected_head_ref=item.expected_head_ref,
    )


def drain_table_actions(
    args: argparse.Namespace,
    kind: str,
    actions: SimpleQueue,
    *,
    build_decision: Callable[[int], Decision | None],
    apply_fn: Callable[[Decision], None],
    fetch: Callable[[int], ApiObject],
    input_queue: SimpleQueue,
    pending: PendingCount,
    forced: set[int],
) -> Callable[[], None]:
    """Poll callback for consume_reviewed: execute operator table actions
    from fairy-ui (y/s/x/r on any row) on this controller thread, where
    the run's args and caches live. ``apply`` posts the persisted review
    (subject to the staleness guard), ``skip``/``cancel`` end the item,
    ``rerun`` forces a fresh LLM pass through the normal pipeline."""

    def poll() -> None:
        while True:
            try:
                number, act = actions.get_nowait()
            except Empty:
                return
            logger.info("table action %r on %s #%s", act, kind, number)
            try:
                if act == "apply":
                    decision = build_decision(number)
                    if decision is None:
                        logger.info(
                            "%s #%s: no reviewed workset file; nothing to apply",
                            kind, number,
                        )
                        continue
                    apply_fn(decision)
                elif act == "skip":
                    workset_transition(args, kind, number, workset.WorkState.SKIPPED)
                elif act == "cancel":
                    workset_transition(args, kind, number, workset.WorkState.CANCELLED)
                elif act == "rerun":
                    # A fresh LLM pass: QUEUED disables verdict reuse and
                    # the forced set bypasses the prepare gates. Fetch
                    # first so a forge error leaves nothing half-done.
                    api = fetch(number)
                    forced.add(number)
                    workset_transition(args, kind, number, workset.WorkState.QUEUED)
                    pending.add(1)
                    input_queue.put(api)
                else:
                    logger.warning("unknown table action %r on %s #%s", act, kind, number)
            except Exception as exc:
                logger.error("table action %r on %s #%s failed: %s", act, kind, number, exc)

    return poll


def workset_transition(
    args: argparse.Namespace, kind: str, number: int, state: workset.WorkState,
) -> None:
    path = workset_path(args, kind, number)
    if path is not None:
        workset.update_item(
            path, lambda item: item.set_state(state, datetime.now(timezone.utc))
        )


def workset_operator_review(
    args: argparse.Namespace, kind: str, decision: Decision,
) -> Decision | None:
    """Re-read the item file just before posting: the file is the
    operator's veto/edit surface. Returns the decision with the file's
    review content (edits to message/labels win; the action mapping is
    not re-derived), or None when the post must not happen (file deleted
    or no longer REVIEWED). Non-LLM decisions never had a file and pass
    through."""
    path = workset_path(args, kind, decision.pr_number)
    if path is None or decision.llm_classification in ("-", ""):
        return decision
    item = workset.load_item(path)
    if item is None:
        logger.info(
            "%s #%d: workset file %s deleted/unreadable; not posting",
            kind, decision.pr_number, path,
        )
        return None
    if item.state is not workset.WorkState.REVIEWED or item.review is None:
        logger.info(
            "%s #%d: workset file is %s, not REVIEWED; not posting",
            kind, decision.pr_number, item.state.name,
        )
        return None
    file_labels = tuple(
        LabelChange(c.label, c.op, c.reason, c.post)
        for c in item.review.label_changes
    )
    if (
        item.review.classification == decision.llm_classification
        and item.review.message == decision.llm_message
        and file_labels == decision.label_changes
    ):
        return decision
    logger.info(
        "%s #%d: posting the operator-edited review from %s",
        kind, decision.pr_number, path,
    )
    return dataclasses_replace(
        decision,
        llm_classification=item.review.classification,
        llm_message=item.review.message,
        label_changes=file_labels,
    )


def workset_prune(args: argparse.Namespace, kind: str, open_numbers: set[int]) -> None:
    """End-of-run cleanup of finished item files. Skipped for
    --forced-only runs: their candidate list is not the open listing,
    so still-open items would look closed."""
    if getattr(args, "forced_only", False):
        return
    d = workset_repo_dir(args)
    if d is None or not d.is_dir():
        return
    workset.prune(
        d, kind, open_numbers,
        datetime.now(timezone.utc) - timedelta(days=args.workset_retention_days),
    )


def workset_on_choice(args: argparse.Namespace, kind: str) -> Callable[[Decision, str], None]:
    """consume_reviewed ``on_choice`` callback: record operator answers."""
    states = {
        "retry": workset.WorkState.QUEUED,
        "skip": workset.WorkState.SKIPPED,
        "cancel": workset.WorkState.CANCELLED,
    }
    return lambda d, choice: workset_transition(args, kind, d.pr_number, states[choice])


class PendingCount:
    """Count of (prepared, decision) tuples still expected on a reviewed
    queue. Thread-safe so a UI thread can add() work while the consumer
    runs."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._lock = Lock()

    def add(self, k: int = 1) -> None:
        with self._lock:
            self._n += k

    @property
    def value(self) -> int:
        with self._lock:
            return self._n


class ReviewUI(Protocol):
    """Interactive frontend for consume_reviewed; None = prompt_manual.

    ``prepared`` is the pipeline's prepared item (PreparedPR /
    issue_fairy.PreparedIssue) or a terminal ``Decision`` for gate skips.
    """

    def candidates(self, items: list[ApiObject]) -> None:
        """Full fetched candidate list, before the pipeline starts."""

    def pipeline(
        self,
        input_queue: SimpleQueue,
        pending: PendingCount,
        cancelled: set[int],
        actions: SimpleQueue,
    ) -> None:
        """Handles for runtime force-add (put + pending.add), cancel, and
        the table-action channel ((number, action) tuples the controller
        executes via its poll_actions callback)."""

    def decide(self, prepared: object, decision: Decision, url: str) -> str:
        """Return "apply", "skip", "defer", "retry", "quit" -- or "hold":
        record the decision and move on without acting; the operator acts
        on the row later through the table-action channel."""

    def item_done(self, prepared: object, decision: Decision) -> None:
        """Called once per item after its final dispatch."""

    def keep_open(self) -> bool:
        """True while the consumer should wait for more work at pending==0."""

    def stopped(self) -> bool:
        """True once the operator asked to quit."""


def consume_reviewed(
    reviewed_queue: SimpleQueue,
    llm_queue: SimpleQueue,
    pending: PendingCount,
    *,
    now: datetime,
    manual: bool,
    approve: bool,
    kind: str,  # "PR" | "issue"; log-line and prompt wording only
    apply: Callable[[object, Decision], None],
    item_url: Callable[[object], str],
    cancelled: set[int] | None = None,
    ui: ReviewUI | None = None,
    on_choice: Callable[[Decision, str], None] | None = None,
    poll_actions: Callable[[], None] | None = None,
) -> tuple[list[Decision], bool]:
    """Drain ``reviewed_queue`` of (prepared, decision) tuples, prompting
    for / applying each actionable decision. ``item_url`` maps a prepared
    item (never a Decision) to its html_url. ``on_choice`` is told each
    operator answer that ends or re-runs an item ("retry", "skip",
    "cancel"). ``poll_actions`` runs every loop iteration (<=0.25s apart
    with a ui) to execute pending operator table actions. Returns the
    collected decisions and whether the operator stopped the run."""
    decisions: list[Decision] = []
    stopped = False
    idle_logged = False
    ready: deque[tuple[object, Decision]] = deque()
    try:
        while True:
            if not ready:
                if pending.value == 0 and (ui is None or not ui.keep_open()):
                    break
                if ui is not None and pending.value == 0 and not idle_logged:
                    idle_logged = True
                    logger.info(
                        "%s side idle: all candidates processed, nothing in "
                        "flight; waiting for force-added work or quit", kind,
                    )
                try:
                    ready.append(reviewed_queue.get(timeout=0.25 if ui else None))
                except Empty:
                    pass
            while True:
                try:
                    ready.append(reviewed_queue.get_nowait())
                except Empty:
                    break
            if poll_actions is not None:
                poll_actions()
            if ui is not None and ui.stopped():
                stopped = True
                break
            if not ready:
                continue

            prepared, d = ready.popleft()
            pending.add(-1)
            idle_logged = False

            age = describe_age(now, d.last_activity)
            prefix = f"{kind} #{d.pr_number}" if d.pr_number >= 0 else f"{kind}<?>"
            action = format_action(d)
            llm_short = format_llm_classification(d.llm_classification)
            needs_interaction = (
                d.action in ACTIONABLE_DECISIONS or decision_has_label_changes(d)
            )
            if needs_interaction and cancelled and d.pr_number in cancelled:
                logger.info("%s #%s: cancelled by operator", kind, d.pr_number)
                needs_interaction = False
                if on_choice is not None:
                    on_choice(d, "cancel")
            # In manual mode the summary line, the LLM note, the labels line
            # and the prompt are one actionable block. Frame it with a blank
            # line before and after -- emitted on stderr, the same stream the
            # log lines and the prompt use -- so the whole block stands out
            # from the SKIP stream and the blanks stay around it instead of
            # landing in the middle.
            interactive = needs_interaction and manual and ui is None
            if interactive:
                sys.stderr.write("\n")
                sys.stderr.flush()
            logger.info(
                f"{prefix}: {action} age={age:>8} author={d.author:<20} "
                f"auto={d.auto_merge:<10} llm={llm_short:<9}  {d.reason}  -- {d.title}"
            )
            if d.llm_message:
                logger.info(f"    LLM: {d.llm_message}")
            for c in d.label_changes:
                logger.info(
                    "    label %s %s%s: %s",
                    c.op, c.label, " (post)" if c.post else "", c.reason or "-",
                )

            if needs_interaction and (manual or approve or ui is not None):
                if ui is not None:
                    choice = ui.decide(
                        prepared, d,
                        "" if isinstance(prepared, Decision) else item_url(prepared),
                    )
                elif manual:
                    choice = prompt_manual(
                        d.pr_number, manual_action_description(d),
                        pr_url="" if isinstance(prepared, Decision)
                        else item_url(prepared),
                    )
                    # Close the framed block (see ``interactive`` above).
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                else:
                    choice = "apply"
                if choice == "retry":
                    if isinstance(prepared, Decision):
                        logger.info(
                            "%s #%s has no queued LLM evaluation to retry",
                            kind, d.pr_number,
                        )
                    else:
                        logger.info("Retrying %s #%s LLM evaluation", kind, d.pr_number)
                        if on_choice is not None:
                            on_choice(d, "retry")
                        pending.add(1)
                        llm_queue.put(prepared)
                    continue
                if choice == "defer":
                    pending.add(1)
                    ready.append((prepared, d))
                    continue
                if choice == "quit":
                    stopped = True
                elif choice == "apply":
                    try:
                        apply(prepared, d)
                    except Exception as exc:
                        logger.error(
                            "ERROR applying %s for %s #%s: %s",
                            manual_action_description(d), kind, d.pr_number, exc,
                        )
                elif choice == "skip" and on_choice is not None:
                    on_choice(d, "skip")

            decisions.append(d)
            if ui is not None:
                ui.item_done(prepared, d)
            if stopped:
                break
    except KeyboardInterrupt:
        stopped = True
        logger.warning("Stopped by user.")
    return decisions, stopped


def start_review_pipeline(
    args: argparse.Namespace,
    input_queue: SimpleQueue,
    *,
    now: datetime,
    self_login: str | None,
    wip_re: re.Pattern[str],
    cache: gcli_cache.Cache,
    discussion_cache_max_age: timedelta,
    cancelled: set[int] | None = None,
) -> tuple[SimpleQueue[tuple[PreparedItem, Decision]], SimpleQueue[PreparedPR | object]]:
    """``input_queue`` feeds PR ApiObjects to the prepare worker until
    ``_PREPARE_DONE``; a UI may keep injecting force-added PRs after
    start. ``cancelled`` numbers skip their queued LLM evaluation."""
    llm_queue: SimpleQueue[PreparedPR | object] = SimpleQueue()
    reviewed_queue: SimpleQueue[tuple[PreparedItem, Decision]] = SimpleQueue()

    def prepare_worker() -> None:
        queued = 0
        try:
            while (pr := input_queue.get()) is not _PREPARE_DONE:
                if args.limit and queued >= args.limit:
                    decision = Decision(
                        pr.get("number", 0),
                        str(pr.get("title") or ""),
                        get_pr_author(pr),
                        "-",
                        "skip",
                        f"candidate not evaluated; --limit {args.limit} reached",
                        first_dt(pr, "updated_at", "created_at"),
                    )
                    reviewed_queue.put((decision, decision))
                    continue
                prepared = safe_prepare_pr(
                    args,
                    pr,
                    now=now,
                    self_login=self_login,
                    wip_re=wip_re,
                    cache=cache,
                    discussion_cache_max_age=discussion_cache_max_age,
                )
                if isinstance(prepared, Decision):
                    reviewed_queue.put((prepared, prepared))
                else:
                    queued += 1
                    workset_record_queued(
                        args, "pr",
                        number=prepared.number,
                        title=prepared.title,
                        html_url=str(prepared.pr.get("html_url") or ""),
                    )
                    llm_queue.put(prepared)
        finally:
            try:
                logger.debug("cache save after prepare phase start path=%s", args.cache)
                gcli_cache.save_cache(args.cache, cache)
                logger.debug("cache save after prepare phase ok path=%s", args.cache)
            except Exception as exc:
                logger.warning("failed to save cache after prepare phase %s: %s", args.cache, exc)

    def llm_worker() -> None:
        while True:
            prepared = llm_queue.get()
            if prepared is _LLM_REVIEW_DONE:
                return
            if not isinstance(prepared, PreparedPR):
                raise RuntimeError(f"unexpected LLM queue item type: {type(prepared)!r}")
            if cancelled and prepared.number in cancelled:
                logger.info(
                    "PR #%s: skipping queued LLM evaluation: cancelled by operator",
                    prepared.number,
                )
                workset_transition(args, "pr", prepared.number, workset.WorkState.CANCELLED)
                reviewed_queue.put((prepared, Decision(
                    prepared.number, prepared.title, prepared.author,
                    prepared.auto_merge, "skip", "cancelled by operator",
                    prepared.last_activity,
                )))
                continue
            decision = safe_apply_llm_review_to_prepared(args, prepared)
            workset_record_reviewed(
                args, "pr", decision,
                expected_updated_at=prepared.pr.get("updated_at"),
                expected_head_ref=get_pr_head_ref(prepared.pr),
            )
            reviewed_queue.put((prepared, decision))

    llm_parallelism = max(1, int(getattr(args, "llm_parallelism", 1) or 1))
    logger.debug("starting review pipeline llm_parallelism=%d", llm_parallelism)

    Thread(target=prepare_worker, name="pr-prepare", daemon=True).start()
    for i in range(llm_parallelism):
        name = "pr-llm" if llm_parallelism == 1 else f"pr-llm-{i + 1}"
        Thread(target=llm_worker, name=name, daemon=True).start()
    return reviewed_queue, llm_queue


def warn_simulate_past_limitations(ignore_after: datetime) -> None:
    """Surface what ``--simulate-past`` does NOT rewrite."""
    logger.warning(
        "--simulate-past=%s active. Limitations:\n"
        "  * CI status: current Forgejo state, not the state at the cutoff.\n"
        "  * Wrapper web_search reaches today's web; use --web-search off\n"
        "    (or cached) in your --llm-review-cmd. (vector_store_search is\n"
        "    fine if --repo-root points at the prepped mirror.)\n"
        "  * PR/comment bodies: post-cutoff edits cannot be reverted.\n"
        "  * Dismissed reviews: cannot be revived.\n"
        "  * --patch-repo (and the wrapper's --repo-root etc.) must be a\n"
        "    cutoff-prepped mirror: master rewound, every replayed PR's\n"
        "    head pinned at --patch-pr-ref-template. The bot trusts those\n"
        "    refs verbatim; nothing here verifies they match the cutoff.\n"
        "  * Pass --cache <separate-path> to keep the live PR-data cache clean,\n"
        "    and --workset-dir <separate-path> to keep the live work set clean.",
        ignore_after.isoformat(),
    )


def run_reviews(args: argparse.Namespace, ui: ReviewUI | None = None) -> int:
    """Fetch candidates, run the review pipeline and drain it. Everything
    main() does after logging setup, so an embedding UI can run the PR
    side on its own thread with an already-parsed args namespace."""
    if args.llm_review_cmd and args.patch_repo is None:
        logger.error("--llm-review-cmd requires --patch-repo PATH")
        return 2
    if args.forced_only and not args.force_review_prs:
        logger.error("--forced-only requires at least one --force-review-pr")
        return 2
    if args.simulate_past is not None:
        if not args.patch_pr_ref_template or "{number}" not in args.patch_pr_ref_template:
            logger.error(
                "--simulate-past requires --patch-pr-ref-template TEMPLATE "
                "containing ``{number}`` (e.g. fforge/pr/{number})."
            )
            return 2
        warn_simulate_past_limitations(args.simulate_past)
    now = datetime.now(timezone.utc)
    wip_prefixes = DEFAULT_WIP_PREFIXES + (args.wip_prefixes or [])
    wip_re = compile_wip_regex(wip_prefixes)
    cache = gcli_cache.load_cache(args.cache)
    discussion_cache_max_age = timedelta(hours=args.discussion_cache_max_age_hours)

    try:
        self_login = get_self_login(args)

        def pr_sort_key(pr: ApiObject) -> tuple[bool, int]:
            number = pr.get("number")
            pr_number = int(number) if str(number).isdigit() else 10**12
            return (pr_number not in args.force_review_prs, pr_number)

        if args.forced_only:
            # Skip the open-PR listing entirely; fetch only the named
            # IDs. Avoids running unrelated LLM reviews in ad-hoc /
            # --simulate-past replays.
            prs = [get_pr(args, n) for n in sorted(args.force_review_prs)]
        else:
            prs = list_open_prs(args)
            # ``list_open_prs`` only returns currently-open PRs, so pull
            # any forced IDs not in the listing by number. Makes
            # ``--force-review-pr`` work for closed/merged PRs.
            listed = {pr["number"] for pr in prs}
            prs += [get_pr(args, n) for n in sorted(args.force_review_prs - listed)]
        prs = sorted(prs, key=pr_sort_key)
    except Exception as exc:
        logger.error("ERROR: %s", exc)
        return 2

    submitted_counts = {action: 0 for action in ACTIONABLE_DECISIONS}

    if ui is not None:
        ui.candidates(prs)
    input_queue: SimpleQueue = SimpleQueue()
    for pr in prs:
        input_queue.put(pr)
    if ui is None:
        input_queue.put(_PREPARE_DONE)
    cancelled: set[int] | None = set() if ui is not None else None
    pending = PendingCount(len(prs))

    reviewed_queue, llm_queue = start_review_pipeline(
        args,
        input_queue,
        now=now,
        self_login=self_login,
        wip_re=wip_re,
        cache=cache,
        discussion_cache_max_age=discussion_cache_max_age,
        cancelled=cancelled,
    )
    poll_actions = None
    if ui is not None:
        table_actions: SimpleQueue = SimpleQueue()
        ui.pipeline(input_queue, pending, cancelled, table_actions)
        poll_actions = drain_table_actions(
            args, "pr", table_actions,
            build_decision=lambda n: workset_table_decision(args, n),
            apply_fn=lambda d: apply_decision(
                args, d, d, cache=cache, submitted_counts=submitted_counts,
            ),
            fetch=lambda n: get_pr(args, n),
            input_queue=input_queue,
            pending=pending,
            forced=args.force_review_prs,
        )

    try:
        decisions, stopped_by_user = consume_reviewed(
            reviewed_queue, llm_queue, pending,
            now=now,
            manual=args.manual,
            approve=args.approve,
            kind="PR",
            apply=lambda prepared, d: apply_decision(
                args, prepared, d, cache=cache, submitted_counts=submitted_counts,
            ),
            item_url=lambda prepared: url
            if isinstance(url := prepared.pr.get("html_url"), str) else "",
            cancelled=cancelled,
            ui=ui,
            on_choice=workset_on_choice(args, "pr"),
            poll_actions=poll_actions,
        )
    finally:
        if ui is not None:
            input_queue.put(_PREPARE_DONE)
        for _ in range(max(1, int(getattr(args, "llm_parallelism", 1) or 1))):
            llm_queue.put(_LLM_REVIEW_DONE)
        try:
            gcli_cache.save_cache(args.cache, cache)
        except Exception as exc:
            logger.warning("failed to save PR-data cache %s: %s", args.cache, exc)

    workset_prune(args, "pr", {pr["number"] for pr in prs})

    auto_counts = Counter(d.auto_merge for d in decisions)
    llm_counts = Counter(d.llm_classification for d in decisions)
    auto_merge_candidates = [d for d in decisions if d.auto_merge == "merge"]
    auto_merge_approvable = [d for d in decisions if would_auto_merge_after_approval(d)]

    matched = sum(1 for d in decisions if d.action == "approve")
    actionable_total = sum(1 for d in decisions if d.action in ACTIONABLE_DECISIONS)
    auto_counts_text = ", ".join(
        f"{name}={auto_counts[name]}"
        for name in sorted(auto_counts, key=lambda x: (x == "?", x))
    )
    llm_counts_text = ", ".join(
        f"{name}={llm_counts[name]}"
        for name in sorted(llm_counts, key=lambda x: (x == "-", x))
        if llm_counts[name]
    )
    logger.info(
        f"\nSummary: {len(decisions)} open PR(s) checked, approve={matched}, actionable={actionable_total}, "
        f"submitted approve={submitted_counts['approve']}, comment={submitted_counts['comment']}, request_changes={submitted_counts['request_changes']}."
    )
    logger.info(f"Auto-merge status counts: {auto_counts_text}")
    logger.info(
        "Scheduled auto-merge PR(s): %d", len(auto_merge_candidates),
    )
    logger.info(
        "Would auto-merge when approved: %d", len(auto_merge_approvable),
    )
    if auto_merge_approvable:
        details = ", ".join(
            f"#{d.pr_number}"
            for d in sorted(auto_merge_approvable, key=lambda x: x.pr_number)
            if d.pr_number >= 0
        )
        logger.info("Auto-merge on approval: %s", details)

    if args.llm_review_cmd:
        logger.info("LLM review classifications: %s", llm_counts_text)

    attention_pr_decisions = sorted(
        (
            d for d in decisions
            if (d.cancelled_ci_contexts or d.blocked_ci_contexts)
            and d.pr_number >= 0
        ),
        key=lambda x: x.pr_number,
    )
    if attention_pr_decisions:
        # Forgejo's HTTP API exposes neither a ``…/runs/{id}/rerun`` nor
        # a way to release required-condition gates (verified against
        # the swagger spec; first observed on Forgejo 15.0.0+gitea-
        # 1.22.0), so we just dump a list per PR for an admin operator
        # to walk through in the UI. CANCELLED and BLOCKED contexts are
        # also already excluded from the LLM CI-triage payload upstream,
        # so the model was not asked to nag about them and no triage
        # tokens were spent on them.
        logger.info(
            "PRs with CI contexts requiring manual action in the Forgejo "
            "UI (%d total):",
            len(attention_pr_decisions),
        )
        for d in attention_pr_decisions:
            parts: list[str] = []
            for state, ctxs in (
                ("CANCELLED", d.cancelled_ci_contexts),
                ("BLOCKED", d.blocked_ci_contexts),
            ):
                if not ctxs:
                    continue
                preview = ", ".join(ctxs[:4])
                if len(ctxs) > 4:
                    preview += f", +{len(ctxs) - 4} more"
                parts.append(f"{state} ({len(ctxs)}): {preview}")
            logger.info("  #%d  %s", d.pr_number, "; ".join(parts))

    merge_ready_decisions = sorted(
        (d for d in decisions if d.merge_ready and d.pr_number >= 0),
        key=lambda x: x.pr_number,
    )
    if merge_ready_decisions:
        # Set only on the "already approved by self" skip path: bot has
        # already approved, ``pr.mergeable`` is True (else the conflicts
        # gate would have caught it), and auto-merge is not queued. So
        # all that's left is for someone to click Merge.
        logger.info(
            "PRs already approved by fairy and waiting to be merged "
            "(%d total):",
            len(merge_ready_decisions),
        )
        for d in merge_ready_decisions:
            logger.info("  #%d  %s", d.pr_number, d.title)

    # External-approver reminder: PRs with a non-bot APPROVED reviewer
    # that fairy is NOT acting on this run (action="skip") and that
    # are NOT already in fairy-approved list above. Disjoint from
    # the merge_ready list by construction.
    externally_approved_decisions = sorted(
        (
            d for d in decisions
            if d.external_approvers
            and not d.merge_ready
            and d.action == "skip"
            and d.pr_number >= 0
        ),
        key=lambda x: x.pr_number,
    )
    if externally_approved_decisions:
        logger.info(
            "PRs approved by an external reviewer and waiting to be "
            "merged (%d total):",
            len(externally_approved_decisions),
        )
        for d in externally_approved_decisions:
            approvers = ", ".join(d.external_approvers)
            logger.info(
                "  #%d  [approved by %s]  %s",
                d.pr_number, approvers, d.title,
            )

    if actionable_total and not args.approve and not args.manual:
        logger.info("Dry-run only. Re-run with --approve to submit actions.")

    return 130 if stopped_by_user else 0


def main() -> int:
    args = parse_args()
    setup_logging(
        logger, args.verbose,
        forge_gcli.logger, gcli_cache.logger, workset.logger, ci_log.logger,
        color=args.color,
    )
    return run_reviews(args)


if __name__ == "__main__":
    raise SystemExit(main())
