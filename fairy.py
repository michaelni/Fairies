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

The PR side of the repo agent: gates, LLM payload, submit seams.
agent.py drives it for tickets and posting, worker.py for reviews;
there is no standalone PR pipeline anymore. parse_args defines the
per-side configuration string the agent/worker/fairy-ui all accept.

Selection rules:
1. PR is open.
2. PR is not marked WIP/draft.
3. PR has no *currently outstanding* request-for-changes review.
4. The PR head commit has CI statuses reported. ERROR/FAILURE jobs do not
   block the LLM pipeline: the failure details and log tails ride along in
   the ``ci_triage`` payload so a full review can analyze the failure.
5. PR has had no activity for at least 7 days.

The LLM review:
A PR that passes the rules is sent to the external reviewer command in
--llm-review-cmd. That command receives JSON on stdin containing a fixed
review prompt, PR metadata, and the patch text. It must return JSON with one of
these classifications:

- approve
- minor_issues_approve
- moderate_issues
- major_issues
- skip

Reads go through `gcli api`, approvals through `gcli pulls ... approve`.
--auto-mode lets the agent's send pass post standing verdicts
by itself; without it every verdict waits in reviewed/ for the operator.

When a human @-mentions fairy or requests it as a reviewer, the run may enter a
"forced review" path, which bypasses the rules above but not the reviewer.

Without ``--llm-review-cmd`` nothing is reviewed and nothing is approved:
a PR that gets as far as the reviewer skips naming the missing command,
and the attention classes (merge-ready, ci-blocked, awaiting-approver)
are all that is left to act on. README-NO-LLM.md describes that
deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict as dataclasses_asdict, dataclass, replace as dataclasses_replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, TypeAlias
from urllib.parse import urlencode, urljoin

import branch_persist
import ci_log
import git_util
import gcli_cache
from common import (
    EXIT_REVIEW_HALTED,
    EXIT_TURN_FAILED,
    parse_turn_failure,
    JsonObject,
    add_color_arg,
    attachment_urls,
    iso_to_dt,
    parse_iso_datetime_arg,
    parse_side_args,
    setup_logging,
)
import forge_gcli
from forge_gcli import (
    AUTO_MERGE_CANCEL_EVENT,
    AUTO_MERGE_SCHEDULE_EVENT,
    PUSH_EVENT,
    add_forge_repo_args,
    apply_issue_label_changes,
    build_repo_path,
    gcli_api,
    gcli_create_pr,
    list_issue_timeline,
    load_json,
    post_issue_comment,
    run_cmd,
)
from forgejo_export import labels
from llm_output_hacks import link_published_branch
import workset

__all__ = [
    "ACTIONABLE_DECISIONS",
    "ApiObject",
    "DEFAULT_WIP_PREFIXES",
    "Decision",
    "LLMReview",
    "LabelChange",
    "REVIEWED_PR_MIN_AGE_DAYS",
    "add_llm_exec_args",
    "add_side_agent_args",
    "add_side_identity_args",
    "apply_triage_labels",
    "build_llm_discussion",
    "call_llm_with_retries",
    "compile_user_mention_regex",
    "compile_wip_regex",
    "decision_from_review",
    "decision_has_label_changes",
    "discussion_time",
    "effective_review_states",
    "first_dt",
    "flatten_label_args",
    "flatten_pr_number_args",
    "format_llm_classification",
    "get_auto_merge_info",
    "get_item_author_login",
    "get_last_activity",
    "get_pr",
    "get_pr_author",
    "get_pr_base_sha",
    "get_pr_head_branch",
    "get_pr_head_ref",
    "get_pr_head_sha",
    "get_pr_thread",
    "invoke_llm_wrapper",
    "item_body_mentions_user",
    "label_names",
    "list_open_prs",
    "list_recently_closed_prs",
    "llm_skip_reason",
    "logger",
    "make_parser",
    "manual_action_description",
    "max_dt",
    "parse_args",
    "post_label_explanations",
    "prepared_pr_from_dict",
    "prepared_to_dict",
    "safe_apply_llm_review_to_prepared",
    "safe_prepare_pr",
    "scan_closed_cutoff",
    "submit_decision_action",
    "validate_sides",
    "validate_worker_sides",
    "with_operator_notes",
]


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


class ReviewHalted(RuntimeError):
    """A review container is suspect; this PR must not be retried."""


class ReviewTurnFailed(RuntimeError):
    """The wrapper spent its in-run retry budget on provider-ended
    turns (``EXIT_TURN_FAILED``): more outer attempts would re-run the
    whole ensemble against the same content flag, so the item goes to
    error/ and the agent's doubling backoff paces the next try."""


@dataclass(frozen=True)
class LLMReview:
    classification: str
    message: str
    label_changes: tuple[LabelChange, ...] = ()
    # Collected branch records (see branch_persist): quarantined branches
    # this review wants published as fairy/<name> on approval.
    branches: tuple[JsonObject, ...] = ()


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
    branches: tuple[JsonObject, ...] = ()
    # For merge_ready: when fairy's approval review was submitted, so
    # the operator surface can show how long the merge has waited.
    approved_at: datetime | None = None


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
    # When the PR head is CI-red, structured failure details (incl. log
    # tails) for the review wrapper. ``None`` for green CI.
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
    # pre-check votes ``skip``. Set on every explicitly forced review
    # (--force-review and operator rerun requests). Travels as the
    # ``ignore_triage_skip`` field in the wrapper's stdin request.
    ignore_triage_skip: bool = False
    # The review was forced (an @-mention, a Forgejo review request, or
    # --force-review): the agent must not drop it at --limit, or a
    # human's explicit ask starves behind stale eligible items forever.
    forced_review: bool = False
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


def prepared_to_dict(prepared) -> dict:
    """JSON-safe dict of a PreparedPR/PreparedIssue for a filedb ticket."""
    data = dataclasses_asdict(prepared)
    if data.get("last_activity") is not None:
        data["last_activity"] = prepared.last_activity.isoformat()
    return data


def prepared_pr_from_dict(data: dict) -> PreparedPR:
    d = dict(data)
    if d.get("last_activity"):
        d["last_activity"] = datetime.fromisoformat(d["last_activity"])
    for key in ("cancelled_ci_contexts", "blocked_ci_contexts", "external_approvers"):
        d[key] = tuple(d.get(key) or ())
    return PreparedPR(**d)

_LLM_REVIEW_DONE = object()


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


def add_side_identity_args(p: argparse.ArgumentParser) -> None:
    """The side options every fairy process needs: the repo identity
    and credentials, logging and the gcli cache."""
    add_forge_repo_args(p)
    p.add_argument(
        "--log-file",
        type=Path,
        help="agent and worker additionally log to this file; fairy-ui "
             "tails it into its logs pane (level-tagged line format)",
    )
    p.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Verbosity level for command logging: "
            "0 = none, "
            "1 = write commands and the initial item listing, "
            "2 = all commands."
        ),
    )
    p.add_argument(
        "--branch-push",
        action="append",
        default=[],
        type=parse_branch_push_spec,
        metavar="NAME=OWNER/REPO=URL",
        help="Enable branch persistence for the container repo NAME "
             "(podman-backed reviewers get it as their fairy remote): a "
             "verdict's fairy/<name> pushes and deletions go to the git "
             "URL, whose default branch is first fast-forwarded to the "
             "forge tip --patch-repo tracks; its pull requests are opened "
             "in OWNER/REPO -- all only when the review is sent (operator "
             "y, or --auto-mode). Repeat per repo.",
    )
    p.add_argument(
        "--branch-push-head-owner",
        action="append",
        default=[],
        metavar="NAME=OWNER",
        help="When NAME's --branch-push URL is a fork: the fork's owner, "
             "so pull requests name OWNER:fairy/<branch> as their head "
             "(default: the OWNER of --branch-push).",
    )
    add_color_arg(p)
    p.add_argument(
        "--cache",
        type=Path,
        help="Pickle cache path holding the side's gcli data (default: "
             "~/.fairy/<forge>_<account>_<owner>_<repo>_<pulls|issues>.pkl). "
             "The issue cache is shared with forgejo_export.py, so issues "
             "fetched by one are reused by the other. Saves are whole-file "
             "last-writer-wins: a concurrent run can discard the other's "
             "fresh entries (refetched later), never corrupt them.",
    )


def add_side_agent_args(p: argparse.ArgumentParser, *,
                        min_age_default: float) -> None:
    """The side options only the agent's scan and send passes read,
    shared by the PR and issue sides; ``min_age_default`` is the
    side's --min-age-days default."""
    p.add_argument(
        "--min-age-days",
        type=float,
        default=min_age_default,
        help=(
            "Minimum age of last discussion activity (in days) before the "
            "item is proactively analyzed, including PRs whose head CI is "
            "red. Does not apply when a "
            "human @-mentions fairy, the item is forced, or fairy is a "
            "requested reviewer. Once fairy has engaged, the effective "
            "threshold drops to at most 6h. "
            "Default: 7 for PRs, 14 for issues."
        ),
    )
    p.add_argument(
        "--auto-mode",
        action="store_true",
        help="The agent's send pass posts standing verdicts on its own. "
             "Without this flag they wait in reviewed/ for the TUI's y "
             "or --ask.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N LLM evaluations (0 = no limit). Caps cost and "
             "provider rate-limit windows; gate-skipped items do not count.",
    )
    p.add_argument(
        "--force-review",
        action="append",
        type=parse_pr_number_csv,
        default=None,
        metavar="N[,N...]",
        help=(
            "Force review (and potential approval) of the specified item "
            "number(s), bypassing the usual selection checks -- for an issue "
            "including the open-state gate; a closed/merged PR additionally "
            "needs --force-review-non-open. Can be repeated or passed as a "
            "comma-separated list."
        ),
    )
    p.add_argument(
        "--force-skip",
        action="append",
        type=parse_pr_number_csv,
        default=None,
        metavar="N[,N...]",
        help=(
            "Always skip the specified item number(s). Can be repeated or "
            "passed as a comma-separated list. Takes precedence over "
            "--force-review."
        ),
    )
    p.add_argument(
        "--forced-only",
        action="store_true",
        help="Limit the run to the items named via --force-review "
             "(no open-item listing). Requires at least one --force-review.",
    )
    p.add_argument(
        "--workset-retention-days",
        type=float,
        default=14.0,
        help="Days a settled filedb ticket (posted/skipped/cancelled/"
             "error) is kept after its last state change; items still "
             "in the open listing are never pruned (default: 14).",
    )
    p.add_argument(
        "--scan-closed-days",
        type=float,
        default=0.0,
        help="Keep the filedb items/ snapshots (state, labels, discussion) "
             "of closed PRs/issues fresh while their last update lies "
             "within this many days -- visibility only: closed items are "
             "never gated, queued or reviewed by this option. Their cached "
             "discussion is refetched only when their updated_at moves; "
             "the --discussion-cache-max-age-hours TTL applies to open "
             "items only. 0 disables (default).",
    )
    p.add_argument(
        "--discussion-cache-max-age-hours",
        type=float,
        default=24.0,
        help="Time-to-live (hours) on the cached discussion (a PR's "
             "comments / reviews / inline review-comments trio, an issue's "
             "comments); these can be silently edited or deleted server-side "
             "without bumping the item's updated_at, the TTL forces a "
             "periodic refetch as a backstop. Other PR fields (timeline, "
             "commits, files) are gated only on pr.updated_at and ignore "
             "this TTL. (default: 24)",
    )


def add_llm_exec_args(p: argparse.ArgumentParser) -> None:
    """The side options the worker's review execution reads, shared by
    the PR and issue sides."""
    p.add_argument(
        "--patch-repo",
        type=Path,
        metavar="PATH",
        help="Local clone of the reviewed repo; the PR side synthesizes "
             "the PR patch from it via git format-patch (required there "
             "with --llm-review-cmd).",
    )
    p.add_argument(
        "--llm-review-cmd",
        help=(
            "External command used to review candidate PRs / analyze "
            "candidate issues (pr_review_wrapper.py; --task issue is "
            "appended on the issue side). It receives JSON on stdin and "
            "must print JSON with classification and message on stdout."
        ),
    )
    p.add_argument(
        "--podman-host",
        action="append",
        default=[],
        metavar="[LABEL=]USER@HOST[,port=N][,cpus=N][,memory=SIZE][,gpu=DEV]",
        help=(
            "Run LLM shell work (review, triage, repro, bisect, ...) in "
            "ephemeral containers on this podman host (passwordless ssh "
            "destination); repeat for more machines, the first being the "
            "default. Each value is forwarded as --shell-host to "
            "--llm-review-cmd, so that command must be the wrapper. "
            "Provision each host first with containers/provision_remote.py."
        ),
    )
    p.add_argument(
        "--llm-timeout",
        type=int,
        default=3600*5,
        help="Timeout in seconds for the external LLM command "
             "(default: 18000)",
    )
    p.add_argument(
        "--llm-max-attempts",
        type=int,
        default=3,
        help=(
            "Maximum number of LLM attempts per item (default: 3). "
            "If the LLM command fails (non-zero exit, timeout, malformed "
            "JSON, or unknown classification) we retry up to this many total "
            "attempts before giving up and recording the item as 'error'."
        ),
    )
    p.add_argument(
        "--llm-retry-delay",
        type=float,
        default=5.0,
        help=(
            "Seconds to sleep between failed LLM attempts (default: 5). "
            "Applied only between attempts; no delay before the first attempt "
            "or after the final one."
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
        "--triage-label",
        action="append",
        type=parse_label_csv,
        dest="triage_label",
        default=None,
        metavar="LABEL[,LABEL...]",
        help=(
            "Label name the triage/review model may add or remove. Can be "
            "repeated or passed as a comma-separated list. Passed to the LLM "
            "command as ``triage_label_allowlist`` in the stdin JSON "
            "payload."
        ),
    )


def _add_pr_agent_args(p: argparse.ArgumentParser) -> None:
    """The PR-only options of the agent's scan and send passes."""
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
        "--force-review-non-open",
        action="store_true",
        help="Let --force-review also review closed/merged PRs. Off by "
             "default: a forced PR whose state is not ``open`` is skipped "
             "with ``not open``. (WIP/draft and conflicting PRs are always "
             "reviewed when forced, independent of this flag.)",
    )
    p.add_argument(
        "--force-engage",
        action="store_true",
        help="When reviewing a PR named via --force-review, run the full "
             "reviewer pass regardless of the triage route (overrides both "
             "``skip`` and ``reply_no_verdict``) and even when the head CI is red. "
             "Only applies to PRs named with --force-review (not @mention / "
             "requested-reviewer engagements).",
    )


def _add_pr_worker_args(p: argparse.ArgumentParser) -> None:
    """The PR-only options of the worker's review execution: the patch
    and payload are built at review time."""
    p.add_argument(
        "--llm-max-patch-bytes",
        type=int,
        default=200000,
        help="Maximum number of patch bytes sent to the LLM (default: 200000)",
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
        "--simulate-past",
        type=parse_iso_datetime_arg,
        metavar="ISO_DATETIME",
        help="Filter cached events to ISO_DATETIME and resolve PR heads "
             "from --patch-repo refs (which the operator preps for the "
             "cutoff). Pair with --patch-pr-ref-template and a separate "
             "--cache. See startup warning for residual limits.",
    )


def make_parser(agent: bool = True, worker: bool = True) -> argparse.ArgumentParser:
    """The PR side's parser: the identity options plus the ``agent`` /
    ``worker`` scopes -- a program registers only the scopes whose
    options it reads."""
    p = argparse.ArgumentParser(
        description="The PR side of the repo agent. Pass these arguments to "
                    "configurator.py, in its --prs section.",
    )
    add_side_identity_args(p)
    if agent:
        add_side_agent_args(p, min_age_default=7.0)
        _add_pr_agent_args(p)
    if worker:
        add_llm_exec_args(p)
        _add_pr_worker_args(p)
    return p


def parse_args(argv: list[str] | None = None, *, agent: bool = True,
               worker: bool = True) -> argparse.Namespace:
    args = parse_side_args(make_parser(agent=agent, worker=worker),
                           None if agent and worker else make_parser(), argv)
    args.cache = args.cache or gcli_cache.side_cache_path(args, "pulls")
    if agent:
        args.force_review_prs = flatten_pr_number_args(args.force_review)
        args.force_skip_prs = flatten_pr_number_args(args.force_skip)
    if worker:
        args.triage_labels = flatten_label_args(args.triage_label)
    return args


def _config_error(message: str) -> None:
    logger.error(message)
    raise SystemExit(2)


def validate_worker_sides(pr_ns: argparse.Namespace | None,
                          issue_ns: argparse.Namespace | None) -> None:
    """Reject broken side configs with rc=2: discovered per-item they
    would burn error retries for days. The worker-scope checks -- the
    configurator and the agent run validate_sides, the worker this."""
    if pr_ns and pr_ns.llm_review_cmd and pr_ns.patch_repo is None:
        _config_error("--llm-review-cmd requires --patch-repo PATH")
    for ns in (pr_ns, issue_ns):
        if ns is not None and getattr(ns, "simulate_past", None) is not None \
                and "{number}" not in (getattr(ns, "patch_pr_ref_template", None) or ""):
            _config_error(
                "--simulate-past requires --patch-pr-ref-template TEMPLATE "
                "containing {number} (e.g. fforge/pr/{number})")


def validate_sides(pr_ns: argparse.Namespace | None,
                   issue_ns: argparse.Namespace | None) -> None:
    """validate_worker_sides plus the agent-scope checks, over full
    side namespaces (the configurator's writes, the agent's reads)."""
    validate_worker_sides(pr_ns, issue_ns)
    for ns, forced in ((pr_ns, "force_review_prs"),
                       (issue_ns, "force_review_issues")):
        if ns is not None and ns.forced_only and not getattr(ns, forced):
            _config_error(
                "--forced-only requires at least one --force-review")


def combine_review_messages(*parts: str) -> str:
    clean = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(clean)


def gcli_approve(args: argparse.Namespace, pr_number: int, review_message: str = "") -> None:
    """Approve the PR with the deployment's --approve-message prepended
    to the review's own message."""
    forge_gcli.gcli_approve(
        args, pr_number,
        combine_review_messages(args.approve_message, review_message))


@dataclass(frozen=True)
class BranchPushSpec:
    """Where one container repo's approved fairy branches go: the git
    push URL, and the forge repo (``owner``/``forge_repo``) its pull
    requests are opened in."""
    repo: str
    owner: str
    forge_repo: str
    url: str


def parse_branch_push_spec(value: str) -> BranchPushSpec:
    """Parse a --branch-push ``NAME=OWNER/REPO=URL`` (the URL may itself
    contain ``=``)."""
    name, sep1, rest = value.partition("=")
    target, sep2, url = rest.partition("=")
    owner, slash, forge_repo = target.partition("/")
    if not (sep1 and sep2 and slash and name and owner and forge_repo and url):
        raise argparse.ArgumentTypeError(
            f"--branch-push {value!r}: expected NAME=OWNER/REPO=URL")
    return BranchPushSpec(repo=name, owner=owner, forge_repo=forge_repo, url=url)


def branch_push_head_owner(args: argparse.Namespace, spec: BranchPushSpec) -> str:
    for entry in args.branch_push_head_owner:
        name, sep, owner = entry.partition("=")
        if sep and name == spec.repo and owner:
            return owner
    return spec.owner


def link_published_branches(
    args: argparse.Namespace,
    message: str,
    branches: tuple[JsonObject, ...],
    ticket_url: str,
) -> str:
    """Link the message's mentions of the branches it pushes, and of
    their tip commits, to their pages in the --branch-push fork on the
    forge serving ``ticket_url``."""
    if not ticket_url:
        return message
    specs = {s.repo: s for s in getattr(args, "branch_push", None) or ()}
    for record in branches:
        spec = specs.get(record["repo"])
        if spec is None or record["mode"] == "delete":
            continue
        owner = branch_push_head_owner(args, spec)
        forge_branch = f"{branch_persist.FAIRY_BRANCH_PREFIX}{record['branch']}"
        message = link_published_branch(
            message, forge_branch, record["sha"],
            forge_gcli.branch_page_url(args, ticket_url, owner, spec.forge_repo, forge_branch),
            forge_gcli.commit_page_url(args, ticket_url, owner, spec.forge_repo, record["sha"]))
    return message


def fast_forward_fork_default_branch(
    args: argparse.Namespace,
    spec: BranchPushSpec,
) -> None:
    """Fast-forward the --branch-push fork's default branch to the tip
    the --patch-repo checkout tracks for it on the forge, so the fork
    follows the reviewed repo. A refused or failed push is logged and
    does not block the send."""
    checkout = args.patch_repo
    if checkout is None:
        return
    try:
        branch = git_util.git_remote_default_branch(checkout, spec.url)
        sha = git_util.git_resolve_first(
            checkout, [f"refs/remotes/{remote}/{branch}"
                       for remote in git_util.FORGE_REMOTES])
        if sha is None:
            raise RuntimeError(f"{checkout} tracks no {branch} of the forge")
        git_util.git_push_refspecs(checkout, spec.url,
                                   [f"{sha}:refs/heads/{branch}"],
                                   force=False, timeout_s=300.0)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        logger.warning("default branch of %s not fast-forwarded: %s",
                       spec.url, exc)
        return
    logger.info("fast-forwarded %s of %s to %s", branch, spec.url, sha[:12])


def publish_decision_branches(
    args: argparse.Namespace,
    decision: Decision,
) -> None:
    """Apply the decision's branch records to their --branch-push
    destinations -- push as ``fairy/<name>``, or delete the published
    branch -- and open the pull requests they request, after
    fast-forwarding each touched fork's default branch. Every push
    precedes the first PR creation, and the first failure raises so the
    caller blocks the whole send with its reason."""
    specs = {s.repo: s for s in args.branch_push}
    unconfigured = {r for record in decision.branches
                    if (r := record.get("repo")) not in specs}
    if unconfigured:
        # checked before the first push: a mid-loop refusal would leave
        # the earlier records published for a send that cannot complete
        raise branch_persist.BranchTransferError(
            f"no --branch-push configured for repo(s) "
            f"{', '.join(sorted(map(repr, unconfigured)))}")
    for spec in {specs[record["repo"]] for record in decision.branches}:
        fast_forward_fork_default_branch(args, spec)
    published: list[tuple[JsonObject, BranchPushSpec, str]] = []
    for record in decision.branches:
        spec = specs[record["repo"]]
        forge_branch = branch_persist.publish_branch_record(
            record, remote_url=spec.url)
        logger.info("#%s: published %s of %s (%s, %s)", decision.pr_number,
                    record["mode"], forge_branch, spec.repo,
                    record["sha"][:12])
        published.append((record, spec, forge_branch))
    for record, spec, forge_branch in published:
        pr = record.get("pr")
        if not isinstance(pr, dict):
            continue
        # Re-pushing the same SHA is a no-op, but re-creating a PR is
        # not: the forge refuses a duplicate open PR from the same head
        # (gcli_create_pr cites the per-forge sources), so a send
        # retried after a partial failure takes that refusal as the PR
        # already being open -- verified against the open listing
        # before the refusal is swallowed.
        try:
            gcli_create_pr(
                args,
                spec.owner,
                spec.forge_repo,
                branch_push_head_owner(args, spec),
                forge_branch,
                str(pr.get("target") or ""),
                str(pr.get("title") or ""),
                str(pr.get("body") or ""),
            )
        except RuntimeError as exc:
            if not any(
                head.get("ref") == forge_branch
                for open_pr in list_open_prs(args, owner=spec.owner,
                                             repo=spec.forge_repo)
                if isinstance(head := open_pr.get("head"), dict)
            ):
                raise
            logger.info("#%s: a pull request from %s is already open; "
                        "create refused with: %s", decision.pr_number,
                        forge_branch, exc)
            continue
        logger.info("#%s: opened pull request from %s into %s of %s/%s",
                    decision.pr_number, forge_branch, pr.get("target"),
                    spec.owner, spec.forge_repo)


def branch_publication_block(
    args: argparse.Namespace, decision: Decision,
) -> str | None:
    """Publish the decision's branch records, shared by the PR and
    issue submit paths; the reason the send must block when they cannot
    all be published, None when there is nothing to publish or
    everything went out."""
    if not decision.branches:
        return None
    if not getattr(args, "branch_push", None):
        return "verdict carries branches but --branch-push is not configured"
    try:
        publish_decision_branches(args, decision)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        logger.error("#%s: branch publication failed: %s",
                     decision.pr_number, exc)
        return f"branch publication failed: {exc}"
    return None


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
    # ``--force-review`` with --force-review-non-open posts even to a
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
    skip_guard: bool = False,
) -> str | None:
    """Post ``decision``; None on success, the staleness guard's block
    reason otherwise. ``skip_guard`` posts without the staleness check:
    the operator's force_post waives it. Branches are published before
    the review message, so the posted text never names a branch that
    failed to appear."""
    changed_reason = None if skip_guard else check_pr_still_unchanged(args, prepared, decision)
    if changed_reason is not None:
        logger.info("PR #%s: SKIP            submit skipped because %s", decision.pr_number, changed_reason)
        return changed_reason
    blocked = branch_publication_block(args, decision)
    if blocked is not None:
        return blocked
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
            gcli_cache.entry_key(args, "pulls", args.owner, args.repo, decision.pr_number),
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
    return None


def first_dt(obj: ApiObject, *keys: str) -> datetime | None:
    for key in keys:
        dt = iso_to_dt(obj.get(key))
        if dt is not None:
            return dt
    return None


def max_dt(values: Iterable[datetime | None]) -> datetime | None:
    return max((v for v in values if v is not None), default=None)


def list_open_prs(args: argparse.Namespace, owner: str | None = None,
                  repo: str | None = None) -> list[ApiObject]:
    query = urlencode({"state": "open", "sort": "leastupdate", "limit": 100})
    path = build_repo_path(owner or args.owner, repo or args.repo,
                           f"/pulls?{query}")
    data = gcli_api(args, path, all_pages=True, verbose_threshold=1)
    if not isinstance(data, list):
        raise RuntimeError(f"expected list of PRs, got {type(data).__name__}")
    return [pr for pr in data if isinstance(pr, dict)]


def scan_closed_cutoff(args: argparse.Namespace) -> datetime | None:
    """The oldest ``updated_at`` --scan-closed-days still covers, from
    the simulated clock under --simulate-past; None when the option is
    off."""
    if args.scan_closed_days <= 0:
        return None
    now = getattr(args, "simulate_past", None) or datetime.now(timezone.utc)
    return now - timedelta(days=args.scan_closed_days)


def list_recently_closed_prs(args: argparse.Namespace) -> list[ApiObject]:
    """Closed PRs inside the --scan-closed-days window; [] when off."""
    cutoff = scan_closed_cutoff(args)
    if cutoff is None:
        return []
    return forge_gcli.list_closed_since(args, "pulls", cutoff)


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


def get_pr_thread(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    cache: gcli_cache.Cache,
    cache_max_age: timedelta,
) -> tuple[list[ApiObject], list[ApiObject], list[ApiObject], list[ApiObject]]:
    """``(reviews, comments, review_comments, timeline)`` for ``pr`` in
    one ``gcli_cache.get``: the four fields share one entry stamp and
    one ``updated_at`` refetch, where separate discussion and timeline
    calls pay two -- and, on a changed PR, drop and refetch each
    other's fields as non-refetched siblings."""
    pr_number = int(pr["number"])
    live_updated_at = iso_to_dt(pr.get("updated_at"))
    if live_updated_at is None:
        raise RuntimeError(
            f"PR #{pr_number} is missing or has unparseable updated_at; "
            f"refusing to cache against an unknown freshness key"
        )
    fields = gcli_cache.get(
        cache, args, "pulls", args.owner, args.repo, pr_number,
        live_updated_at, "reviews", "issue_comments", "review_comments",
        "timeline", max_age=cache_max_age,
    )
    return (
        list(fields["reviews"]),
        list(fields["issue_comments"]),
        list(fields["review_comments"]),
        list(fields["timeline"]),
    )


def get_pr_timeline(
    args: argparse.Namespace,
    pr: ApiObject,
    *,
    cache: gcli_cache.Cache,
    cache_max_age: timedelta,
) -> list[ApiObject]:
    """Timeline events for ``pr`` via the cache; [] when ``updated_at``
    is unusable. A failed fetch raises -- the caller decides whether
    the timeline is optional."""
    live_updated_at = iso_to_dt(pr.get("updated_at"))
    if live_updated_at is None:
        return []
    fields = gcli_cache.get(
        cache, args, "pulls", args.owner, args.repo, int(pr["number"]),
        live_updated_at, "timeline", max_age=cache_max_age,
    )
    return list(fields["timeline"])


def list_commit_statuses(args: argparse.Namespace, ref: str) -> list[ApiObject]:
    """CI rows for ``ref``, less anything --simulate-past puts in the future."""
    return filter_activity_after(
        forge_gcli.list_commit_statuses(args, args.owner, args.repo, ref),
        getattr(args, "simulate_past", None), "created_at", "updated_at",
    )


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

    Wraps ``normalize_status_state`` on the row's ``state`` and adds
    two description-based overrides for Forgejo Actions:

    - ``FAILURE`` / ``ERROR`` rows whose description matches
      ``_CANCELLED_DESCRIPTION_RE`` are reclassified as ``CANCELLED``.
    - ``PENDING`` rows whose description matches
      ``_BLOCKED_DESCRIPTION_RE`` are reclassified as ``BLOCKED``.

    This is the only place that knows about either Forgejo Actions
    shape; downstream helpers that need to ask "is this row
    cancelled / blocked / failing / pending" should call this rather
    than ``normalize_status_state`` directly.
    """
    state = normalize_status_state(row.get("state"))
    description = row.get("description") or ""
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
        context = status.get("context")
        if not isinstance(context, str) or not context:
            continue
        state = row_effective_state(status)
        if state is None:
            continue
        when = first_dt(status, "updated_at", "created_at")
        result[context] = (state, when)
    return result


# Commit contexts whose latest state is one of these enter the
# ``ci_triage`` payload. PENDING / NEUTRAL etc. do not: we still skip
# those PRs early without calling the LLM (same as before, but without
# treating them as "failure" notifications).
CI_TRIAGE_NAG_STATES: frozenset[str] = frozenset({"FAILURE", "ERROR"})


def _status_row_time(st: ApiObject) -> datetime:
    return first_dt(st, "created_at", "updated_at") or datetime.min.replace(tzinfo=timezone.utc)


def group_commit_statuses_by_context(statuses: list[ApiObject]) -> dict[str, list[ApiObject]]:
    groups: dict[str, list[ApiObject]] = {}
    for st in statuses:
        if not isinstance(st, dict):
            continue
        context = st.get("context")
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
        desc = last.get("description") or ""
        target = last.get("target_url") or ""
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


def build_ci_triage_payload(
    head_sha: str,
    failure_details: list[dict[str, object]],
) -> JsonObject:
    return {
        "head_sha": head_sha,
        "failure_contexts": failure_details,
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


def get_pr_head_sha(pr: ApiObject) -> str | None:
    """The PR's head commit SHA; unlike get_pr_head_ref there is no
    branch-name fallback, so the result is usable as a git revision
    pin."""
    head = pr.get("head")
    candidates = [head.get("sha")] if isinstance(head, dict) else []
    candidates += [pr.get("sha"), pr.get("head_sha")]
    return next((v for v in candidates if isinstance(v, str) and v), None)


def get_pr_head_branch(pr: ApiObject) -> str | None:
    head = pr.get("head")
    if isinstance(head, dict) and isinstance(head.get("ref"), str):
        return head["ref"] or None
    return None


def get_pr_base_sha(pr: ApiObject) -> str | None:
    """Base commit of the PR's patch: the forge-computed merge base
    when present, else the base branch tip -- the same choice
    patch_shas_for_run makes for a live review."""
    merge_base = pr.get("merge_base")
    if isinstance(merge_base, str) and merge_base:
        return merge_base
    base = pr.get("base")
    if isinstance(base, dict) and isinstance(base.get("sha"), str):
        return base["sha"] or None
    return None


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
    return forge_gcli.auto_merge_state(args, pr, timeline)


def push_events_from_timeline(timeline: list[ApiObject]) -> list[DiscussionItem]:
    """Extract ``pull_push`` timeline entries as discussion-list items.

    One event is emitted per push to the PR head branch. We surface
    them as synthetic ``kind="push"`` items so the triage and main
    reviewer prompts can see "after my last comment, the author pushed
    commit X" as a first-class signal, instead of having to infer it
    from a SHA mentioned in a prior fairy comment vs the current
    ``head_sha`` (which the triager has been observed to speculate
    around -- see PR #23197 in fairy's history).

    A push whose payload the forge layer could not read carries no
    ``commit_ids`` and is skipped: suppressing it beats blocking the
    whole review on a hard failure.
    """
    items: list[DiscussionItem] = []
    for entry in timeline:
        if entry.get("type") != PUSH_EVENT or "commit_ids" not in entry:
            continue
        commit_ids = entry["commit_ids"]
        user = entry.get("user") or {}
        items.append({
            "kind": "push",
            "author": user.get("login") or user.get("full_name") or "?",
            "created_at": entry.get("created_at"),
            "head_sha": commit_ids[-1] if commit_ids else None,
            "is_force_push": entry.get("is_force_push"),
            "commit_count": len(commit_ids),
        })
    return items

def review_request_events_from_timeline(
        timeline: list[ApiObject]) -> list[DiscussionItem]:
    """``review_request`` timeline entries as ``kind="review_request"``
    discussion items: who asked whom for a review, when, and whether
    the request was withdrawn (``removed``). Valuable to the operator
    and the LLM alike -- a review request IS the invitation the
    reviewer acts on, and it was previously invisible in the
    discussion."""
    items: list[DiscussionItem] = []
    for entry in timeline:
        if entry.get("type") != forge_gcli.REVIEW_REQUEST_EVENT:
            continue
        user = entry.get("user") or {}
        assignee = entry.get("assignee") or {}
        items.append({
            "kind": "review_request",
            "author": user.get("login") or user.get("full_name") or "?",
            "reviewer": assignee.get("login") or assignee.get("full_name")
            or "?",
            "removed": bool(entry.get("removed_assignee")),
            "created_at": entry.get("created_at"),
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


def parse_branch_records(raw: object) -> tuple[JsonObject, ...]:
    """Parse ``branches`` records from wrapper stdout or a ticket.

    Kept lenient on purpose: only entries whose branch, mode, repo and
    sha are non-empty strings survive (the bundle is empty for a
    deletion), and ``branch_persist`` re-checks every field again at
    publication time -- the authoritative boundary, since tickets are
    hand-editable.
    """
    if not isinstance(raw, list):
        return ()
    return tuple(
        item for item in raw
        if isinstance(item, dict) and all(
            isinstance(item.get(key), str) and item[key]
            for key in ("branch", "mode", "repo", "sha"))
        and isinstance(item.get("bundle"), str)
    )


def label_names(changes: tuple[LabelChange, ...], op: str) -> tuple[str, ...]:
    return tuple(c.label for c in changes if c.op == op)


def decision_has_label_changes(decision: Decision) -> bool:
    return bool(decision.label_changes)


def manual_action_description(decision: Decision) -> str:
    parts: list[str] = []
    if decision.action in ACTIONABLE_DECISIONS:
        parts.append(decision.action.replace("_", "-"))
        if decision.action == "approve" and decision.auto_merge == "merge":
            # the y is heavier here: auto-merge is scheduled, so the
            # approval merges the PR, not just comments on it
            parts.append("auto-merge scheduled: approving MERGES the PR")
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
    if decision.branches:
        deletions = sum(1 for b in decision.branches
                        if b.get("mode") == "delete")
        pr_count = sum(1 for b in decision.branches
                       if isinstance(b.get("pr"), dict))
        branch_parts = ([f"push {len(decision.branches) - deletions} branch(es)"]
                        if len(decision.branches) > deletions else []) \
            + ([f"delete {deletions} branch(es)"] if deletions else []) \
            + ([f"open {pr_count} PR(s)"] if pr_count else [])
        parts.append(", ".join(branch_parts))
    return " + ".join(parts) if parts else decision.action


def apply_triage_labels(
    args: argparse.Namespace,
    prepared: PreparedItem,
    decision: Decision,
    *,
    skip_guard: bool,
) -> str | None:
    """Apply label changes for ``decision``; None when applied, the
    staleness guard's block reason when it suppressed them.

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
            return changed_reason

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
    return None


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

    ``timeline`` is the raw ``/issues/{n}/timeline`` payload; its
    ``pull_push`` and ``review_request`` entries are consumed (see
    ``push_events_from_timeline`` /
    ``review_request_events_from_timeline``).
    Passing ``None`` (or omitting it) yields a comments-only discussion.
    The push items carry ``kind="push"`` plus ``head_sha`` /
    ``is_force_push`` so the triage LLM can tell unambiguously that
    new code arrived after its prior comment instead of guessing from
    a SHA mentioned in the comment body. A review without a body is
    kept when it carries a verdict (APPROVED / CHANGES_REQUESTED):
    a bare approve click is discussion-worthy for the operator and
    the LLM alike.
    """
    items: list[DiscussionItem] = []
    if timeline:
        items.extend(push_events_from_timeline(timeline))
        items.extend(review_request_events_from_timeline(timeline))

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
            if normalize_review_state(review.get("state")) \
                    not in ("APPROVED", "CHANGES_REQUESTED"):
                continue
            body = ""
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

    items.sort(key=discussion_time)
    return items


def discussion_time(item: DiscussionItem) -> datetime:
    """A discussion entry's arrival time: forges bump the item's
    updated_at watermark on arrivals, never on edits (see gcli_cache),
    so an entry compares by creation, not by when it was last edited.
    The epoch for an entry without a stamp."""
    return first_dt(item, "submitted_at", "created_at", "updated_at") \
        or datetime.min.replace(tzinfo=timezone.utc)


def with_operator_notes(discussion: list[DiscussionItem],
                        notes: dict | None) -> list[DiscussionItem]:
    """``discussion`` with the entries of the item's notes/ ticket
    merged in at their place in time."""
    return sorted(discussion + ((notes or {}).get("discussion") or []),
                  key=discussion_time)


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
    """A review right after a push races the mirror's fetch schedule:
    the new head is not locally reachable yet, so a failed format-patch
    is answered by one fetch of the patch repo and a second try
    (production: pr #23903 stayed unretryable for hours)."""
    try:
        data = git_util.git_format_patch_series(args.patch_repo, base_sha, head_sha)
    except RuntimeError:
        git_util.git_fetch_all(args.patch_repo)
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
    ticket_url: str = "",
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
        # a filedb worker points the wrapper at its claimed ticket
        ws_path = getattr(args, "workset_file_override", None)
        if ws_path is not None:
            cmd += [f"--workset-file={ws_path}"]
    if getattr(args, "branch_push", None):
        cmd += ["--persist-branches"]
        cmd += [f"--persist-repo={spec.repo}" for spec in args.branch_push]
    if extra_cmd_args:
        cmd += list(extra_cmd_args)
    stderr_prefix = f"[wrapper {stderr_tag}=#{number}] " if number is not None else "[wrapper] "
    cp = run_cmd(
        cmd,
        verbose=args.verbose,
        verbose_threshold=2,
        env={**os.environ, "FAIRY_LOG_WIRE": "1"},
        input_text=json.dumps(payload),
        timeout=args.llm_timeout,
        stderr_line_prefix=stderr_prefix,
    )
    if cp.returncode == EXIT_REVIEW_HALTED:
        raise ReviewHalted(
            "LLM review halted: the wrapper flagged a review container as "
            "suspect; inspect the paused container and the debug dumps"
        )
    if cp.returncode == EXIT_TURN_FAILED:
        raise ReviewTurnFailed(
            "LLM review gave up: provider-ended turns exhausted the "
            f"wrapper's in-run retry budget: {parse_turn_failure(cp.stdout)}"
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
    branches = parse_branch_records(data.get("branches"))
    return LLMReview(
        classification=classification,
        message=link_published_branches(args, message.strip(), branches, ticket_url),
        label_changes=label_changes,
        branches=branches,
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
    when every attempt failed; ``ReviewHalted`` and ``ReviewTurnFailed``
    immediately: a PR that left a container suspect must not be run
    again, and a spent turn-failure budget makes more attempts of the
    whole ensemble pointless.
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
        except (KeyboardInterrupt, ReviewHalted, ReviewTurnFailed):
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
        ticket_url=str(pr.get("html_url") or ""),
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
        return Decision(
            number,
            title,
            author,
            auto_merge,
            "skip",
            f"LLM review failed: {exc}",
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
    return f"LLM skip: {first}"[:400] if first else "LLM chose skip"


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
    verdict_kwargs = {"label_changes": review.label_changes,
                      "branches": review.branches}
    if review.classification == "approve":
        return Decision(
            number, title, author, auto_merge, "approve", reason, last_activity,
            review.classification, review.message, **verdict_kwargs,
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
            **verdict_kwargs,
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
            **verdict_kwargs,
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
            **verdict_kwargs,
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
            **verdict_kwargs,
        )
    if review.classification == "skip":
        # a skip only ever applies labels, so its branches would sit in
        # the archive claiming a publication that cannot happen
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
            label_changes=review.label_changes,
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

    # ``--force-review`` means "review this PR no matter what".
    # The WIP/mergeable gates below are heuristics for the cron mode
    # -- when the operator named a PR explicitly we honor that and let
    # the LLM look at it. The closed/merged gate is the exception:
    # reviewing a non-open PR is opt-in via --force-review-non-open,
    # since the usual intent of forcing is a still-open PR.
    # ``--force-skip`` still wins (the first gate below) per its
    # documented precedence.
    is_forced = number in args.force_review_prs

    # Timeline fetches go through ``gcli_cache.get``: it serves a
    # cached copy when ``pr.updated_at`` matches and transparently
    # refetches when it advances, so both consumers (auto-merge
    # detection via ``auto_merge_state_from_timeline``, LLM-discussion
    # enrichment via ``push_events_from_timeline``) see the same single
    # live copy without an extra in-process memo.
    def get_timeline() -> list[ApiObject]:
        try:
            return get_pr_timeline(args, pr, cache=cache,
                                   cache_max_age=discussion_cache_max_age)
        except Exception as exc:
            logger.warning(
                "LLM discussion enrichment: failed to fetch timeline for "
                "PR #%d: %s", number, exc,
            )
            return []

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
        last_activity_value: datetime | None,
        cancelled_ci_contexts: tuple[str, ...] = (),
        blocked_ci_contexts: tuple[str, ...] = (),
        merge_ready: bool = False,
        approved_at: datetime | None = None,
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
            approved_at=approved_at,
        )

    reviews, comments, review_comments = get_pr_discussion(
        args,
        pr,
        cache=cache,
        cache_max_age=discussion_cache_max_age,
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
    # (FFmpeg #22961 and the wider cohort sat skipped after silent
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

    if number in args.force_skip_prs:
        return skip("forced skip by --force-skip", last_activity_value=last_activity)

    if pr.get("state") != "open" and not (is_forced and args.force_review_non_open):
        return skip("not open", last_activity_value=last_activity)

    if is_marked_wip(pr, wip_re) and not is_forced:
        return skip("marked WIP/draft", last_activity_value=last_activity)

    if not pr.get("mergeable") and not is_forced:
        return skip("has conflicts with the target branch",
                    last_activity_value=last_activity)

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

    forced_review_reason: str | None = None
    if number in args.force_review_prs:
        forced_review_reason = "forced review by --force-review"
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
                ci_triage_payload = build_ci_triage_payload(head_ref_for_ci, fd)
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
            ignore_triage_skip=is_forced,
            force_engage=is_forced and args.force_engage,
            forced_review=True,
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
                approved_at=self_state.when,
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
    # at anyway (just a bit later). Not cached: commit-status rows were
    # observed arriving without bumping ``pr.updated_at`` (a Forgejo
    # instance, 2026-08-23, 2 PRs: status rows 34min and 2 days younger
    # than an unmoving updated_at), so the cache's (pr.updated_at + TTL)
    # pattern would serve stale CI -- and per CONTRIBUTING.md a cache
    # that can lie is worse than no cache. A pure-TTL cache is also
    # off the table for the same reason.
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
    # An LLM review proceeds without CI evidence (a no-CI repo never gets
    # a status); with no reviewer command the absent statuses are the
    # more precise skip reason than the rules the PR went on to match.
    if not commit_statuses and not args.llm_review_cmd:
        return skip(
            "no commit statuses / CI results found",
            last_activity_value=last_activity,
            cancelled_ci_contexts=cancelled_ctxs,
            blocked_ci_contexts=blocked_ctxs,
        )

    failing_contexts = sorted(
        ctx for ctx, (state, _) in commit_statuses.items()
        if state not in ("SUCCESS", "NEUTRAL")
    )
    # ``CANCELLED`` contexts are still in ``failing_contexts`` (they are
    # not ``SUCCESS``), so the PR is correctly NOT auto-approved when only
    # cancelled jobs exist. ``NEUTRAL`` is excluded: it is the state of a
    # job that did not run (GitHub reports a conditional or path-filtered
    # job as ``skipped``, and most Actions workflows have one), and
    # treating "did not run" as "not successful" would skip every such PR
    # with no triage payload to explain why.
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
        if not args.llm_review_cmd:
            # Red CI is reviewable only through the LLM pipeline: with no
            # reviewer command there is nobody to hand ci_triage to, and
            # a queued ticket no worker can claim would sit forever.
            return skip(
                f"CI not successful: {preview}",
                last_activity_value=last_activity,
                cancelled_ci_contexts=cancelled_ctxs,
                blocked_ci_contexts=blocked_ctxs,
            )
        ci_payload = build_ci_triage_payload(
            get_pr_head_ref(pr) or "", failure_details
        )
        attach_ci_failure_logs(args, failure_details)
        red_names = [str(d.get("context") or "?") for d in failure_details]
        base_reason = (
            f"CI red ({len(red_names)} ERROR/FAILURE job(s)): "
            f"{', '.join(red_names[:6])}"
            f"{'...' if len(red_names) > 6 else ''}"
        )
        return PreparedPR(
            pr=pr,
            number=number,
            title=title,
            author=author,
            auto_merge=get_auto_merge(),
            last_activity=last_activity,
            base_reason=base_reason,
            discussion=build_llm_discussion(reviews, comments, review_comments, get_timeline()),
            reviewer_username=self_login,
            ci_triage=ci_payload,
            cancelled_ci_contexts=cancelled_ctxs,
            blocked_ci_contexts=blocked_ctxs,
            external_approvers=external_approvers,
        )

    base_reason = f"matches all rules; CI contexts={len(commit_statuses)}"
    if not args.llm_review_cmd:
        return skip(f"{base_reason}; --llm-review-cmd not set (no approval)",
                    last_activity_value=last_activity)

    return PreparedPR(
        pr=pr,
        number=number,
        title=title,
        author=author,
        auto_merge=get_auto_merge(),
        last_activity=last_activity,
        base_reason=base_reason,
        discussion=build_llm_discussion(reviews, comments, review_comments, get_timeline()),
        reviewer_username=self_login,
        external_approvers=external_approvers,
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
        decision = apply_llm_review_to_prepared(args, prepared)
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
