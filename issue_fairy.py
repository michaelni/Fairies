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

Issue-investigator orchestrator: run the LLM wrapper (``--task issue``) on open
issues and apply its verdict (comment + label changes) via gcli.

What belongs here: issue discovery, gating, the issue LLM payload, and
decision application. The gates are label-driven -- the issue's forge
labels are the analysis state machine: resolution/* means done,
repro/* marks a completed analysis pass, "needs info" means waiting on
a human response; a bot @-mention or force flag bypasses them. What
does NOT belong: PR review orchestration (fairy.py) and prompt/schema
definitions (llm_prompt.py / llm_review_api.py).

Like fairy.py this runs dry by default; pass --approve to submit or
--manual to confirm each action. Uses its own gcli cache and bot-state
files so it can run concurrently with fairy.py.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
from urllib.parse import urlencode

import bot_state
import gcli_cache
from common import add_color_arg, default_cache_path, iso_to_dt, setup_logging
import forge_gcli
from forge_gcli import (
    KIND_ISSUE,
    add_forge_repo_args,
    apply_issue_label_changes,
    build_repo_path,
    gcli_api,
    post_issue_comment,
)
from forgejo_export import labels
from llm_review_api import ISSUE_REPORT_CLASSIFICATIONS
import fairy
from fairy import (
    ACTIONABLE_DECISIONS,
    ApiObject,
    Decision,
    LLMReview,
    backoff_for_consecutive_skips,
    build_llm_discussion,
    call_llm_with_retries,
    compile_user_mention_regex,
    compute_llm_skip_backoff,
    decision_has_label_changes,
    describe_age,
    first_dt,
    flatten_label_args,
    flatten_pr_number_args,
    get_item_author_login,
    get_last_activity,
    get_pr_author,
    get_self_login,
    invoke_llm_wrapper,
    item_body_mentions_user,
    label_names,
    list_open_prs,
    manual_action_description,
    max_dt,
    parse_label_csv,
    parse_pr_number_csv,
    post_label_explanations,
    prompt_manual,
    writeback_llm_skip_backoff,
)

__all__ = ["main"]

logger = fairy.logger


@dataclass(frozen=True)
class PreparedIssue:
    """An issue that passed every gate and is ready for the LLM."""
    issue: ApiObject
    number: int
    title: str
    author: str
    last_activity: datetime | None
    base_reason: str
    discussion: list[ApiObject]
    reviewer_username: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze open Forgejo/Gitea issues via an LLM wrapper and post verdicts.",
    )
    add_forge_repo_args(p)
    p.add_argument(
        "--min-age-days",
        type=float,
        default=14.0,
        help=(
            "Minimum age of last discussion activity (in days) before the "
            "issue is proactively analyzed. Does not apply when a human "
            "@-mentions the bot or the issue is forced; once fairy has "
            "engaged on an issue the effective threshold drops to 6h. "
            "Default: 14."
        ),
    )
    p.add_argument(
        "--approve",
        action="store_true",
        help="Actually submit comments/labels. Without this flag the script only reports matches.",
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help="Interactively confirm each matching action.",
    )
    p.add_argument(
        "--llm-review-cmd",
        help=(
            "External command used to analyze candidate issues. It receives JSON "
            "on stdin and must print JSON with classification and message on "
            "stdout (pr_review_wrapper.py; --task issue is appended here)."
        ),
    )
    p.add_argument(
        "--podman-host",
        metavar="USER@HOST",
        default=None,
        help=(
            "Run LLM shell work (repro, bisect, ...) in an ephemeral container "
            "on this podman host (passwordless ssh destination). Appends "
            "--podman --podman-ssh-dest to --llm-review-cmd."
        ),
    )
    p.add_argument(
        "--llm-timeout",
        type=int,
        default=3600 * 5,
        help="Timeout in seconds for the external LLM command (default: 18000)",
    )
    p.add_argument(
        "--llm-parallelism",
        type=int,
        default=1,
        help="Number of LLM subprocesses to run in parallel (default: 1).",
    )
    p.add_argument(
        "--llm-max-attempts",
        type=int,
        default=3,
        help="Maximum number of LLM attempts per issue (default: 3).",
    )
    p.add_argument(
        "--llm-retry-delay",
        type=float,
        default=5.0,
        help="Seconds to sleep between failed LLM attempts (default: 5).",
    )
    p.add_argument(
        "--issue-label",
        action="append",
        type=parse_label_csv,
        dest="issue_label",
        default=None,
        metavar="LABEL[,LABEL...]",
        help=(
            "Label name the LLM may add or remove. Can be repeated or passed "
            "as a comma-separated list. Passed to the LLM command as "
            "``triage_label_allowlist`` in the stdin JSON payload."
        ),
    )
    p.add_argument(
        "--force-review-issue",
        action="append",
        type=parse_pr_number_csv,
        default=None,
        metavar="N[,N...]",
        help=(
            "Force analysis for the specified issue number(s), bypassing the "
            "usual selection checks (including the open-state gate). Can be "
            "repeated or passed as a comma-separated list."
        ),
    )
    p.add_argument(
        "--force-skip-issue",
        action="append",
        type=parse_pr_number_csv,
        default=None,
        metavar="N[,N...]",
        help=(
            "Always skip the specified issue number(s). Takes precedence over "
            "--force-review-issue."
        ),
    )
    p.add_argument(
        "--forced-only",
        action="store_true",
        help="Limit the run to issues named via --force-review-issue "
             "(no open-issue listing). Requires at least one --force-review-issue.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N LLM evaluations (0 = no limit). Caps cost on "
             "test runs; gate-skipped issues do not count.",
    )
    p.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Verbosity level for command logging: 0 = none, 1 = commands and listing, 2 = all.",
    )
    add_color_arg(p)
    p.add_argument(
        "--cache",
        type=Path,
        default=default_cache_path("issue_data_cache.pkl"),
        help="Pickle cache path holding per-issue gcli data "
             "(default: ~/.fairy/issue_data_cache.pkl).",
    )
    p.add_argument(
        "--fairy-state-cache",
        type=Path,
        default=default_cache_path("issue_bot_state.pkl"),
        help="Pickle path for the LLM-skip backoff bookkeeping "
             "(default: ~/.fairy/issue_bot_state.pkl).",
    )
    p.add_argument(
        "--discussion-cache-max-age-hours",
        type=float,
        default=24.0,
        help="Time-to-live (hours) on the cached issue comments; comments can "
             "be edited server-side without bumping issue.updated_at, the TTL "
             "forces a periodic refetch as a backstop (default: 24).",
    )
    args = p.parse_args()
    args.force_review_issues = flatten_pr_number_args(args.force_review_issue)
    args.force_skip_issues = flatten_pr_number_args(args.force_skip_issue)
    args.triage_labels = flatten_label_args(args.issue_label)
    return args


def list_open_issues(args: argparse.Namespace) -> list[ApiObject]:
    """Open real issues: the ``/issues`` listing minus open PRs.

    Forgejo/Gitea and GitHub both surface every PR as an issue in the
    ``/issues`` listing; subtracting the open-PR numbers (the same
    approach forgejo_export.py uses) keeps this forge-generic.
    """
    query = urlencode({"state": "open", "sort": "leastupdate", "limit": 100})
    path = build_repo_path(args.owner, args.repo, f"/issues?{query}")
    data = gcli_api(args, path, all_pages=True, verbose_threshold=1)
    if not isinstance(data, list):
        raise RuntimeError(f"expected list of issues, got {type(data).__name__}")
    pr_numbers = {pr["number"] for pr in list_open_prs(args)}
    issues = [
        item for item in data
        if isinstance(item, dict) and item.get("number") not in pr_numbers
    ]
    logger.debug(
        "issue listing: %d item(s), %d after removing open PRs",
        len(data), len(issues),
    )
    return issues


def get_issue(args: argparse.Namespace, number: int) -> ApiObject:
    path = build_repo_path(args.owner, args.repo, f"/issues/{number}")
    data = gcli_api(args, path)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected issue object for #{number}, got {type(data).__name__}")
    return data


def get_issue_discussion(
    args: argparse.Namespace,
    issue: ApiObject,
    *,
    cache: gcli_cache.Cache,
    cache_max_age: timedelta,
) -> tuple[list[ApiObject], list[ApiObject]]:
    """Return ``(comments, timeline)`` for ``issue`` via the cache."""
    number = int(issue["number"])
    live_updated_at = iso_to_dt(issue.get("updated_at"))
    if live_updated_at is None:
        raise RuntimeError(
            f"issue #{number} is missing or has unparseable updated_at; "
            f"refusing to cache against an unknown freshness key"
        )
    fields = gcli_cache.get(
        cache, args, "issues", args.owner, args.repo, number, live_updated_at,
        "issue_comments", "timeline",
        max_age=cache_max_age,
    )
    return list(fields["issue_comments"]), list(fields["timeline"])


def prepare_issue(
    args: argparse.Namespace,
    issue: ApiObject,
    *,
    now: datetime,
    self_login: str | None,
    cache: gcli_cache.Cache,
    state: bot_state.State,
    discussion_cache_max_age: timedelta,
) -> Decision | PreparedIssue:
    number = int(issue["number"])
    title = str(issue.get("title") or "")
    author = get_pr_author(issue)
    issue_last_activity = first_dt(issue, "updated_at", "created_at")

    def skip(reason: str, last_activity: datetime | None = issue_last_activity) -> Decision:
        return Decision(number, title, author, "-", "skip", reason, last_activity)

    if number in args.force_skip_issues:
        return skip("forced skip by --force-skip-issue")

    # A forced issue is analyzed no matter what, including closed ones
    # (e.g. re-checking a fixed regression); everything below the force
    # check is a cron-mode heuristic.
    is_forced = number in args.force_review_issues

    if issue.get("state") != "open" and not is_forced:
        return skip("not open")

    comments, timeline = get_issue_discussion(
        args, issue, cache=cache, cache_max_age=discussion_cache_max_age,
    )

    # Like fairy.py, last_activity deliberately comes from the discussion
    # only, not ``issue.updated_at``: label/milestone/assignee edits bump
    # updated_at but are not conversation the bot should react to. The
    # issue body itself counts via created_at (a fresh issue with no
    # comments is still activity).
    last_activity = get_last_activity(
        issue, [], comments, [], include_pr_updated=False,
    )
    self_last = get_last_activity(
        issue, [], comments, [],
        predicate=lambda item: get_item_author_login(item) == self_login,
    ) if self_login else None

    forced_reason: str | None = None
    if is_forced:
        forced_reason = "forced review by --force-review-issue"
    elif self_login:
        mention_re = compile_user_mention_regex(self_login)
        latest_mention = get_last_activity(
            issue, [], comments, [],
            predicate=lambda item: (
                get_item_author_login(item) != self_login
                and item_body_mentions_user(item, mention_re)
            ),
        )
        if author != self_login and item_body_mentions_user(issue, mention_re):
            latest_mention = max_dt(
                [latest_mention, first_dt(issue, "created_at")]
            )
        if latest_mention is not None and (self_last is None or latest_mention > self_last):
            forced_reason = f"later discussion mentions {self_login}"

    if forced_reason is None:
        # The issue's labels are the analysis state machine (set by the
        # LLM's own label_changes or by humans); a mention or force
        # bypasses it, everything else is gated on them.
        issue_labels = set(labels(issue))
        resolutions = sorted(l for l in issue_labels if l.startswith("resolution/"))
        if resolutions:
            return skip(f"resolved: {resolutions[0]}", last_activity)
        if "needs info" not in issue_labels and any(
            l.startswith("repro/") for l in issue_labels
        ):
            # A repro/* label marks a completed full pass (duplicate,
            # regression and root-cause checks included). Removing it or
            # mentioning the bot re-triggers analysis. When "needs info"
            # is also set the analysis is explicitly unfinished, so the
            # waiting gate below decides instead.
            return skip("already analyzed: repro/* set", last_activity)
        last_nonself = get_last_activity(
            issue, [], comments, [],
            predicate=lambda item: get_item_author_login(item) != self_login,
        )
        if self_last is not None and (last_nonself is None or last_nonself <= self_last):
            return skip("no non-bot activity since fairy's last reply", last_activity)
        if last_activity is None:
            return skip("cannot determine activity timestamp", None)
        # Once fairy has engaged on the issue, a human response only
        # needs to settle briefly (same constant as re-reviewed PRs);
        # fresh issues wait the full --min-age-days.
        min_age_days = float(args.min_age_days)
        if self_last is not None:
            min_age_days = min(min_age_days, fairy.REVIEWED_PR_MIN_AGE_DAYS)
        if last_activity > now - timedelta(days=min_age_days):
            return skip("activity is newer than threshold", last_activity)
        # Issues have no head SHA; the backoff key degenerates to
        # last_activity alone, which is exactly the "anything new was
        # said" bypass we want.
        entry = state.entries.get(bot_state.Key(args.owner, args.repo, number), {})
        backoff_info = compute_llm_skip_backoff(entry, None, last_activity, now)
        if backoff_info is not None:
            consec, eligible_at = backoff_info
            return skip(
                f"in LLM skip-backoff window after {consec} consecutive skip(s); "
                f"window={backoff_for_consecutive_skips(consec)}; "
                f"next eligible at {eligible_at.isoformat()}",
                last_activity,
            )

    if not args.llm_review_cmd:
        return skip(
            f"{forced_reason or 'candidate for analysis'}; --llm-review-cmd not set",
            last_activity,
        )

    return PreparedIssue(
        issue=issue,
        number=number,
        title=title,
        author=author,
        last_activity=last_activity,
        base_reason=forced_reason or "stale enough for analysis",
        discussion=build_llm_discussion([], comments, [], timeline),
        reviewer_username=self_login,
    )


def run_llm_issue(
    args: argparse.Namespace,
    prepared: PreparedIssue,
    extra_cmd_args: list[str] | None,
) -> LLMReview:
    issue = prepared.issue
    payload: dict[str, object] = {
        "issue": {
            "number": prepared.number,
            "title": prepared.title,
            "body": issue.get("body") or "",
            "author": prepared.author,
            "html_url": issue.get("html_url") or "",
            "created_at": issue.get("created_at") or "",
            "state": issue.get("state") or "",
            "labels": labels(issue),
        },
        "discussion": prepared.discussion,
        "reviewer_username": prepared.reviewer_username or "",
    }
    if args.triage_labels:
        payload["triage_label_allowlist"] = args.triage_labels
    return invoke_llm_wrapper(
        args,
        payload,
        number=prepared.number,
        allowed_classifications=frozenset(ISSUE_REPORT_CLASSIFICATIONS),
        label_allowlist=args.triage_labels,
        extra_cmd_args=["--task", "issue"] + list(extra_cmd_args or []),
        stderr_tag="issue",
    )


def evaluate_issue(args: argparse.Namespace, prepared: PreparedIssue) -> Decision:
    """LLM verdict -> Decision: ``reply`` posts the message as an issue
    comment, ``skip`` posts nothing; label changes apply either way."""
    try:
        review = call_llm_with_retries(
            args,
            prepared.number,
            lambda extra: run_llm_issue(args, prepared, extra),
        )
    except Exception as exc:
        max_attempts = max(1, int(args.llm_max_attempts or 1))
        return Decision(
            prepared.number, prepared.title, prepared.author, "-", "skip",
            f"LLM analysis failed after {max_attempts} attempt(s): {exc}",
            prepared.last_activity, "error", "",
        )
    action = "comment" if review.classification == "reply" else "skip"
    reason = (
        "LLM chose skip" if review.classification == "skip"
        else f"{prepared.base_reason}; LLM: {review.classification}"
    )
    return Decision(
        prepared.number, prepared.title, prepared.author, "-", action, reason,
        prepared.last_activity, review.classification, review.message,
        expected_pr_updated_at=prepared.issue.get("updated_at"),
        label_changes=review.label_changes,
    )


def check_issue_still_unchanged(
    args: argparse.Namespace, decision: Decision,
) -> str | None:
    current = get_issue(args, decision.pr_number)
    if current.get("state") != "open" and decision.pr_number not in args.force_review_issues:
        return "issue is no longer open"
    if (
        decision.expected_pr_updated_at is not None
        and current.get("updated_at") != decision.expected_pr_updated_at
    ):
        return "issue updated_at changed"
    return None


def apply_issue_decision(
    args: argparse.Namespace,
    decision: Decision,
    *,
    cache: gcli_cache.Cache,
    submitted_counts: dict[str, int],
) -> None:
    """Post the comment (if any), then apply label changes.

    The staleness guard runs once, before the comment, on pristine
    ``updated_at``; the comment itself bumps updated_at so labels are
    applied without re-checking (same ordering as fairy's
    ``apply_decision``).
    """
    changed_reason = check_issue_still_unchanged(args, decision)
    if changed_reason is not None:
        logger.info(
            "issue #%s: SKIP            submit skipped because %s",
            decision.pr_number, changed_reason,
        )
        return
    if decision.action == "comment":
        post_issue_comment(
            args, args.owner, args.repo, decision.pr_number,
            decision.llm_message, kind=KIND_ISSUE,
        )
        submitted_counts["comment"] += 1
        # Same eventual-consistency race as fairy's submit path: drop the
        # cached entry so the next run refetches the comment list.
        cache.entries.pop(
            gcli_cache.EntryKey("issues", args.owner, args.repo, decision.pr_number),
            None,
        )
        try:
            gcli_cache.save_cache(args.cache, cache)
        except Exception as exc:
            logger.warning(
                "failed to persist cache invalidation after commenting on "
                "issue #%s: %s", decision.pr_number, exc,
            )
    if decision_has_label_changes(decision):
        current = set(labels(get_issue(args, decision.pr_number)))
        apply_issue_label_changes(
            args,
            args.owner,
            args.repo,
            decision.pr_number,
            list(label_names(decision.label_changes, "add")),
            list(label_names(decision.label_changes, "remove")),
            current,
            kind=KIND_ISSUE,
        )
        post_label_explanations(
            args, decision.pr_number, decision.label_changes, current,
            kind=KIND_ISSUE,
        )


_LLM_DONE = object()


def start_issue_pipeline(
    args: argparse.Namespace,
    issues: list[ApiObject],
    *,
    now: datetime,
    self_login: str | None,
    cache: gcli_cache.Cache,
    state: bot_state.State,
    discussion_cache_max_age: timedelta,
) -> tuple[
    SimpleQueue[tuple[PreparedIssue | Decision, Decision]],
    SimpleQueue[PreparedIssue | object],
]:
    """One prepare thread feeds ``--llm-parallelism`` LLM workers; every
    issue yields exactly one ``(prepared, decision)`` on the reviewed
    queue. ``--limit`` caps how many candidates reach the LLM."""
    llm_queue: SimpleQueue[PreparedIssue | object] = SimpleQueue()
    reviewed_queue: SimpleQueue[tuple[PreparedIssue | Decision, Decision]] = SimpleQueue()

    def prepare_worker() -> None:
        queued = 0
        for issue in issues:
            try:
                prepared = prepare_issue(
                    args, issue,
                    now=now,
                    self_login=self_login,
                    cache=cache,
                    state=state,
                    discussion_cache_max_age=discussion_cache_max_age,
                )
            except Exception as exc:
                number = issue.get("number")
                logger.error("issue #%s: prepare failed: %s", number, exc)
                prepared = Decision(
                    int(number) if str(number).isdigit() else -1,
                    str(issue.get("title") or ""), get_pr_author(issue),
                    "-", "skip", f"prepare failed: {exc}", None, "error", "",
                )
            if isinstance(prepared, Decision):
                reviewed_queue.put((prepared, prepared))
            elif args.limit and queued >= args.limit:
                reviewed_queue.put((prepared, Decision(
                    prepared.number, prepared.title, prepared.author, "-",
                    "skip", f"candidate not evaluated; --limit {args.limit} reached",
                    prepared.last_activity, "-", "",
                )))
            else:
                queued += 1
                llm_queue.put(prepared)

    def llm_worker() -> None:
        while True:
            prepared = llm_queue.get()
            if prepared is _LLM_DONE:
                return
            if not isinstance(prepared, PreparedIssue):
                raise RuntimeError(f"unexpected LLM queue item type: {type(prepared)!r}")
            d = evaluate_issue(args, prepared)
            writeback_llm_skip_backoff(
                state, args.owner, args.repo, d.pr_number,
                None, prepared.last_activity, d.llm_classification,
            )
            reviewed_queue.put((prepared, d))

    llm_parallelism = max(1, args.llm_parallelism)
    logger.debug("starting issue pipeline llm_parallelism=%d", llm_parallelism)
    Thread(target=prepare_worker, name="issue-prepare", daemon=True).start()
    for i in range(llm_parallelism):
        name = "issue-llm" if llm_parallelism == 1 else f"issue-llm-{i + 1}"
        Thread(target=llm_worker, name=name, daemon=True).start()
    return reviewed_queue, llm_queue


def main() -> int:
    args = parse_args()
    setup_logging(
        logger, args.verbose,
        forge_gcli.logger, gcli_cache.logger, bot_state.logger,
        color=args.color,
    )
    if args.forced_only and not args.force_review_issues:
        logger.error("--forced-only requires at least one --force-review-issue")
        return 2
    now = datetime.now(timezone.utc)
    cache = gcli_cache.load_cache(args.cache)
    state = bot_state.load(args.fairy_state_cache)
    discussion_cache_max_age = timedelta(hours=args.discussion_cache_max_age_hours)

    try:
        self_login = get_self_login(args)
        if args.forced_only:
            issues = [get_issue(args, n) for n in sorted(args.force_review_issues)]
        else:
            issues = list_open_issues(args)
            listed = {issue["number"] for issue in issues}
            issues += [
                get_issue(args, n)
                for n in sorted(args.force_review_issues - listed)
            ]
        issues.sort(key=lambda i: (int(i["number"]) not in args.force_review_issues, int(i["number"])))
    except Exception as exc:
        logger.error("ERROR: %s", exc)
        return 2

    decisions: list[Decision] = []
    llm_counts: dict[str, int] = {}
    submitted_counts = {"comment": 0}
    stopped_by_user = False

    reviewed_queue, llm_queue = start_issue_pipeline(
        args, issues,
        now=now,
        self_login=self_login,
        cache=cache,
        state=state,
        discussion_cache_max_age=discussion_cache_max_age,
    )
    ready: deque[tuple[PreparedIssue | Decision, Decision]] = deque()
    pending = len(issues)
    try:
        while not stopped_by_user:
            if not ready:
                if pending == 0:
                    break
                ready.append(reviewed_queue.get())
            while True:
                try:
                    ready.append(reviewed_queue.get_nowait())
                except Empty:
                    break

            prepared, d = ready.popleft()
            pending -= 1

            age = describe_age(now, d.last_activity)
            logger.info(
                f"issue #{d.pr_number}: {d.action.upper():<15} age={age:>8} "
                f"author={d.author:<20} llm={d.llm_classification:<9}  {d.reason}  -- {d.title}"
            )
            if d.llm_message:
                logger.info(f"    LLM: {d.llm_message}")
            for c in d.label_changes:
                logger.info(
                    "    label %s %s%s: %s",
                    c.op, c.label, " (post)" if c.post else "", c.reason or "-",
                )

            if d.action in ACTIONABLE_DECISIONS or decision_has_label_changes(d):
                if args.manual:
                    url = ""
                    # ``prepared`` is a ``PreparedIssue | Decision`` union
                    # (same shape as fairy's reviewed pipeline): only the
                    # PreparedIssue side carries the issue and can be
                    # re-queued for an LLM retry.
                    if not isinstance(prepared, Decision):
                        raw_url = prepared.issue.get("html_url")
                        if isinstance(raw_url, str):
                            url = raw_url
                    choice = prompt_manual(
                        d.pr_number, manual_action_description(d), pr_url=url,
                    )
                    if choice == "retry":
                        if isinstance(prepared, Decision):
                            logger.info(
                                "issue #%s has no queued LLM evaluation to retry",
                                d.pr_number,
                            )
                        else:
                            pending += 1
                            llm_queue.put(prepared)
                        continue
                    if choice == "defer":
                        # Compensate the ``pending -= 1`` above: the item
                        # goes back in flight.
                        pending += 1
                        ready.append((prepared, d))
                        continue
                    if choice == "quit":
                        stopped_by_user = True
                    elif choice == "apply":
                        try:
                            apply_issue_decision(
                                args, d, cache=cache,
                                submitted_counts=submitted_counts,
                            )
                        except Exception as exc:
                            logger.error(
                                "ERROR applying %s for issue #%s: %s",
                                manual_action_description(d), d.pr_number, exc,
                            )
                elif args.approve:
                    try:
                        apply_issue_decision(
                            args, d, cache=cache,
                            submitted_counts=submitted_counts,
                        )
                    except Exception as exc:
                        logger.error(
                            "ERROR applying %s for issue #%s: %s",
                            manual_action_description(d), d.pr_number, exc,
                        )

            decisions.append(d)
            llm_counts[d.llm_classification] = llm_counts.get(d.llm_classification, 0) + 1
    except KeyboardInterrupt:
        stopped_by_user = True
        logger.warning("Stopped by user.")
    finally:
        for _ in range(max(1, args.llm_parallelism)):
            llm_queue.put(_LLM_DONE)
        try:
            gcli_cache.save_cache(args.cache, cache)
            bot_state.save(args.fairy_state_cache, state)
        except Exception as exc:
            logger.warning("failed to save issue-data cache %s: %s", args.cache, exc)

    actionable_total = sum(1 for d in decisions if d.action in ACTIONABLE_DECISIONS)
    llm_counts_text = ", ".join(
        f"{name}={llm_counts[name]}"
        for name in sorted(llm_counts, key=lambda x: (x == "-", x))
        if llm_counts[name]
    )
    logger.info(
        f"\nSummary: {len(decisions)} issue(s) checked, actionable={actionable_total}, "
        f"submitted comment={submitted_counts['comment']}."
    )
    if args.llm_review_cmd:
        logger.info("LLM issue classifications: %s", llm_counts_text)
    if actionable_total and not args.approve and not args.manual:
        logger.info("Dry-run only. Re-run with --approve to submit actions.")
    return 130 if stopped_by_user else 0


if __name__ == "__main__":
    raise SystemExit(main())
