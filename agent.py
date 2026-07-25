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

The repo agent: one process per repository, PRs and issues together.

Each scan pass lists the forge, runs the gates, and routes every
candidate into a filedb state: gate outcomes become small tickets in
skipped/ci-blocked/merge-ready/awaiting-approver, gate survivors get a
full LLM payload in queued/ -- capped by the per-kind --limit, which
counts exactly the tickets entering queued/. LLM skip verdicts wait
out a doubling backoff in skipped/ and re-queue with the doubled
backoff in the ticket. Standing reviewed/ verdicts with matching
guards suppress re-review (reuse). Operator requests (requests/) are
agent-mediated so the UI never needs forge access; they bypass gates
and the limit. The agent also reaps dead workers' claims, cancels
tickets whose item left the open listing, and prunes settled tickets.

A send pass follows each scan: every outgoing/ item is re-read under
its claim lock, guard-checked against the live forge and posted
through the same submit seams the old pipelines use. A guard failure
returns the verdict to reviewed/ with a note (manual mode) or, under
--approve auto mode -- which itself promotes actionable reviewed/
verdicts to outgoing/ -- re-gates it via skipped/ so a cron run never
stalls. --dry-run logs what would be posted and posts nothing.

What belongs here: the scan pass, the ticket routing policy and the
send pass.
What does NOT belong: file atomicity (filedb), gates and payload
building (fairy / issue_fairy), LLM work (the worker), the UI
(fairy_tui).
"""

from __future__ import annotations

import argparse
import logging
import shlex
import time
from threading import Event
from dataclasses import replace as dataclasses_replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fairy
import filedb
import gcli_cache
import issue_fairy
import workset
from common import (add_file_log, default_cache_path, iso_to_dt,
                    setup_logging, watch_paths)

__all__ = ["main", "scan_pass", "send_pass"]

logger = logging.getLogger(__name__)

MIN_BACKOFF_H = 24.0
ERROR_RETRY_H = 24.0
# States the agent never touches during a scan: the item is being
# worked on or awaits the operator/sender.
IN_FLIGHT = ("queued", "llm", "outgoing")


def backoff_wait_h(prior_backoff_h: float) -> float:
    return max(MIN_BACKOFF_H, 2.0 * float(prior_backoff_h or 0))


def _age_h(data: dict, now: datetime) -> float:
    try:
        changed = datetime.fromisoformat(data["state_changed_at"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return (now - changed).total_seconds() / 3600.0


def gate_state(decision: fairy.Decision) -> str:
    """Attention classes get their own directories; the rest is a plain
    gate skip."""
    if decision.merge_ready:
        return "merge-ready"
    if decision.cancelled_ci_contexts or decision.blocked_ci_contexts:
        return "ci-blocked"
    if decision.external_approvers:
        return "awaiting-approver"
    return "skipped"


def gate_ticket(decision: fairy.Decision, item: dict) -> dict:
    return {
        "title": decision.title,
        "author": decision.author,
        "action": decision.action,
        "reason": decision.reason,
        # so an operator x (-> cancelled) sticks until the item changes
        "expected_updated_at": item.get("updated_at"),
        "cancelled_ci_contexts": list(decision.cancelled_ci_contexts),
        "blocked_ci_contexts": list(decision.blocked_ci_contexts),
        "external_approvers": list(decision.external_approvers),
    }


def _route(db: filedb.Db, kind: str, number: int, state: str, data: dict) -> None:
    """Scan-time routing: dst-first, and refused when the item went
    in-flight (a worker claim, or an operator y moving it to outgoing/)
    during the seconds the prepare took -- the earlier IN_FLIGHT check
    is stale by then and a blind pop would delete the pending work."""
    if not db.replace(state, kind, number, data, unless=IN_FLIGHT):
        logger.info("%s #%d went in-flight while preparing; not rerouted",
                    kind, number)


def scan_side(db: filedb.Db, ns: argparse.Namespace, kind: str, *,
              now: datetime, cache, self_login, forced: set[int]) -> set[tuple[str, int]]:
    """One gate pass over the side's open items; returns the open set."""
    if kind == "pr":
        fetch_one, list_open, forced_ns = fairy.get_pr, fairy.list_open_prs, \
            ns.force_review_prs
        wip_re = fairy.compile_wip_regex(
            fairy.DEFAULT_WIP_PREFIXES + (ns.wip_prefixes or []))
    else:
        fetch_one, list_open, forced_ns = issue_fairy.get_issue, \
            issue_fairy.list_open_issues, ns.force_review_issues
    if ns.forced_only:  # --forced-only: no open listing, just the named items
        items = []
        missing = sorted(forced_ns | forced)
    else:
        items = list_open(ns)
        listed = {int(i["number"]) for i in items
                  if str(i.get("number")).isdigit()}
        # forced/requested numbers may be closed or merged: absent from
        # the open listing but explicitly asked for
        missing = sorted((forced_ns | forced) - listed)
    for n in missing:
        try:
            items.append(fetch_one(ns, n))
        except Exception as exc:
            logger.error("%s #%d: forced fetch failed: %s", kind, n, exc)
            # an error ticket marks the request consumed and puts the
            # failure on screen; an existing ticket already does both
            # (find() reporting the request file itself counts as none)
            if db.find(kind, n) in (None, "requests"):
                db.push("error", kind, n,
                        {"error": f"forced fetch failed: {exc}"})
    # Request-forced numbers bypass the gates through the same ns set
    # the gates read; the addition is undone after the pass so a
    # request does not force every future scan.
    added_forced = forced - forced_ns
    forced_ns |= added_forced
    cache_age = timedelta(hours=ns.discussion_cache_max_age_hours)
    items = [i for i in items if str(i.get("number")).isdigit()]
    open_set = {(kind, int(i["number"])) for i in items}
    open_set |= {(kind, n) for n in forced_ns}

    limit = int(getattr(ns, "limit", 0) or 0)
    try:
        _scan_items(db, ns, kind, items, now=now, cache=cache,
                    self_login=self_login, forced_ns=forced_ns,
                    wip_re=wip_re if kind == "pr" else None,
                    cache_age=cache_age, limit=limit)
    finally:
        forced_ns -= added_forced
    return open_set


def _scan_items(db, ns, kind, items, *, now, cache, self_login, forced_ns,
                wip_re, cache_age, limit) -> None:
    queued = 0
    for item in sorted(items, key=lambda i: (int(i["number"]) not in forced_ns,
                                             int(i["number"]))):
        number = int(item["number"])
        prior = db.find(kind, number)
        if prior in IN_FLIGHT:
            continue
        prior_data = db.get(prior, kind, number) if prior else None
        if prior == "reviewed" and number not in forced_ns and prior_data:
            # Any guard-matching verdict stands -- including skips that
            # carry label changes: they sit in reviewed/ awaiting the
            # operator, and a re-queue would burn an LLM run and yank
            # the row out from under the cursor every scan.
            if (prior_data.get("expected_updated_at") == item.get("updated_at")
                    and (kind != "pr" or prior_data.get("expected_head_ref")
                         == fairy.get_pr_head_ref(item))):
                continue  # standing verdict; reuse
        if prior == "cancelled" and number not in forced_ns and prior_data \
                and prior_data.get("expected_updated_at") == item.get("updated_at"):
            continue  # the operator threw it out; only new activity revives it
        backoff_h = 0.0
        if prior == "skipped" and prior_data and prior_data.get("llm_at"):
            # An LLM skip serves its doubling backoff in skipped/; the
            # file is the memory, so it must not be refreshed early.
            # New activity bypasses the wait outright: a push, comment
            # or @-mention must reach the gates now, not in days (the
            # old compute_llm_skip_backoff keyed on exactly this).
            unchanged = (
                prior_data.get("expected_updated_at") == item.get("updated_at")
                and (kind != "pr" or prior_data.get("expected_head_ref")
                     == fairy.get_pr_head_ref(item)))
            wait = backoff_wait_h(prior_data.get("skip_backoff_h", 0))
            if unchanged and number not in forced_ns \
                    and _age_h(prior_data, now) < wait:
                continue
            # a changed item re-enters without doubling: the doubling
            # counts served waits, not bypasses
            backoff_h = wait if unchanged \
                else float(prior_data.get("skip_backoff_h") or 0)
        if prior == "error" and prior_data and number not in forced_ns \
                and _age_h(prior_data, now) < ERROR_RETRY_H:
            continue  # a persistently failing item must not burn spend every cycle
        if kind == "pr":
            prepared = fairy.safe_prepare_pr(
                ns, item, now=now, self_login=self_login, wip_re=wip_re,
                cache=cache, discussion_cache_max_age=cache_age)
        else:
            try:
                prepared = issue_fairy.prepare_issue(
                    ns, item, now=now, self_login=self_login,
                    cache=cache, discussion_cache_max_age=cache_age)
            except Exception as exc:
                logger.error("issue #%d: prepare failed: %s", number, exc)
                continue
        if isinstance(prepared, fairy.Decision):
            state = gate_state(prepared)
            # A plain gate skip must not clobber the archive: posted/
            # cancelled/error records outrank "nothing to do today".
            if state == "skipped" and prior in ("posted", "cancelled", "error",
                                                "skipped"):
                if prior != "skipped" or (prior_data or {}).get("llm_at"):
                    continue
            _route(db, kind, number, state, gate_ticket(prepared, item))
            continue
        if limit and queued >= limit and number not in forced_ns:
            continue  # nothing written: --limit never persists a skip
        ticket = {
            "title": prepared.title,
            "author": prepared.author,
            "html_url": str(getattr(prepared, "pr", getattr(prepared, "issue", {})).get("html_url") or ""),
            "skip_backoff_h": backoff_h,
            "forced": number in forced_ns,
            "prepared": fairy.prepared_to_dict(prepared),
        }
        _route(db, kind, number, "queued", ticket)
        queued += 1
        logger.info("%s #%d queued (backoff %gh, %d/%s)", kind, number,
                    backoff_h, queued, limit or "inf")


def consume_requests(db: filedb.Db) -> dict[str, set[int]]:
    """Requests force a fresh gate-bypassing ticket; they are deleted
    only after the ticket exists (at-least-once)."""
    forced: dict[str, set[int]] = {"pr": set(), "issue": set()}
    for kind, number in db.list_state("requests"):
        forced[kind].add(number)
    return forced


def finish_requests(db: filedb.Db, forced: dict[str, set[int]]) -> None:
    """Drop the requests this pass consumed, but only once some ticket
    exists for the item (at-least-once: a crashed pass retries). A
    request that arrived mid-pass is not in ``forced`` and waits."""
    for kind, numbers in forced.items():
        for number in numbers:
            # find() would report the request file itself
            if db.find(kind, number) not in (None, "requests"):
                db.pop("requests", kind, number)


def cancel_closed(db: filedb.Db, open_set: set[tuple[str, int]],
                  kinds: set[str]) -> None:
    for state in ("queued", "reviewed"):
        for kind, number in db.list_state(state):
            if kind in kinds and (kind, number) not in open_set:
                db.move(state, "cancelled", kind, number,
                        mutate=lambda d: d.update(reason="not open"))
                logger.info("%s #%d cancelled: left the open listing", kind, number)


def scan_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
              issue_ns: argparse.Namespace | None,
              now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    forced = consume_requests(db)
    open_set: set[tuple[str, int]] = set()
    kinds: set[str] = set()
    # A --forced-only side's open_set is just the named items, not the
    # open listing: everything else would look closed, so closing and
    # pruning are skipped for it (the old workset_prune had the same
    # guard).
    full_kinds: set[str] = set()
    for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
        if ns is None:
            continue
        kinds.add(kind)
        if not ns.forced_only:
            full_kinds.add(kind)
        cache = gcli_cache.load_cache(ns.cache)
        try:
            self_login = fairy.get_self_login(ns)
            open_set |= scan_side(db, ns, kind, now=now, cache=cache,
                                  self_login=self_login, forced=forced[kind])
        finally:
            gcli_cache.save_cache(ns.cache, cache)
    finish_requests(db, forced)
    cancel_closed(db, open_set, full_kinds)
    for kind, number in db.reap():
        logger.warning("%s #%d re-queued: its worker died", kind, number)
    if full_kinds == kinds:  # prune's keep-set is kind-blind
        retention_ns = pr_ns or issue_ns
        before = now - timedelta(days=retention_ns.workset_retention_days)
        for state in ("posted", "skipped", "cancelled", "error"):
            db.prune(state, before, keep=open_set)


def ticket_decision(kind: str, number: int, ticket: dict) -> fairy.Decision | None:
    """Rebuild a postable Decision purely from a verdict ticket; the
    ticket's guard rides on the Decision so the staleness checks pin
    the post to the reviewed state."""
    review = ticket.get("review") or {}
    if not review.get("classification"):
        return None
    llm = fairy.LLMReview(
        classification=review["classification"],
        message=review.get("message", ""),
        label_changes=tuple(fairy.LabelChange(**c)
                            for c in review.get("label_changes") or ()))
    if kind == "pr":
        decision = fairy.decision_from_review(
            llm, number=number, title=ticket.get("title", ""),
            author=ticket.get("author", ""), auto_merge="-",
            last_activity=iso_to_dt(ticket.get("last_activity_iso")),
            base_reason="persisted review")
    else:
        decision = issue_fairy.issue_review_decision(
            llm, number=number, title=ticket.get("title", ""),
            author=ticket.get("author", ""), reason="persisted review",
            last_activity=iso_to_dt(ticket.get("last_activity_iso")))
    return dataclasses_replace(
        decision,
        expected_pr_updated_at=ticket.get("expected_updated_at"),
        expected_head_ref=ticket.get("expected_head_ref"))


def postable(decision: fairy.Decision | None) -> bool:
    return decision is not None and (
        decision.action in fairy.ACTIONABLE_DECISIONS
        or fairy.decision_has_label_changes(decision))


def post_decision(ns: argparse.Namespace, kind: str, decision: fairy.Decision,
                  *, cache, counts: dict[str, int]) -> bool:
    """The forge side effects, through the same seams the old pipelines
    post through; False when the staleness guard blocked the post."""
    if kind == "issue":
        return issue_fairy.submit_issue_decision(
            ns, decision, cache=cache, submitted_counts=counts)
    if decision.action in fairy.ACTIONABLE_DECISIONS:
        if not fairy.submit_decision_action(ns, decision, decision, cache=cache):
            return False
        counts[decision.action] += 1
        if fairy.decision_has_label_changes(decision):
            fairy.apply_triage_labels(ns, decision, decision, skip_guard=True)
        return True
    return fairy.apply_triage_labels(ns, decision, decision, skip_guard=False)


def send_one(db: filedb.Db, ns: argparse.Namespace, kind: str, number: int, *,
             cache, counts: dict[str, int], dry_run: bool) -> str | None:
    """Post one outgoing/ item under its claim lock; returns the state
    it ended in (None: not claimed, or dry run)."""
    claim = db.claim("outgoing", "outgoing", kind, number)
    if claim is None:
        return None
    try:
        ticket = claim.read()  # last-moment read: operator edits count
        decision = ticket_decision(kind, number, ticket)
        if not postable(decision):
            claim.finish("reviewed", dict(ticket, send_blocked="nothing to post"))
            return "reviewed"
        if dry_run:
            logger.info("%s #%d: DRY RUN, would post: %s", kind, number,
                        fairy.manual_action_description(decision))
            claim.abort()
            return None
        if kind == "pr":
            reason = fairy.check_pr_still_unchanged(ns, decision, decision)
        else:
            reason = issue_fairy.check_issue_still_unchanged(ns, decision)
        if reason is None:
            if post_decision(ns, kind, decision, cache=cache, counts=counts):
                claim.finish("posted", dict(
                    ticket, posted_at=datetime.now(timezone.utc).isoformat()))
                logger.info("%s #%d posted: %s", kind, number,
                            fairy.manual_action_description(decision))
                return "posted"
            reason = "item changed during submit"
        if getattr(ns, "approve", False):
            # Auto mode must not stall on a stale verdict: without
            # llm_at the skipped/ ticket is re-gated (and, the item
            # having changed, freshly re-reviewed) on the next scan.
            ticket.pop("llm_at", None)
            ticket["skip_backoff_h"] = 0
            state = "skipped"
        else:
            state = "reviewed"
        claim.finish(state, dict(ticket, send_blocked=reason))
        logger.info("%s #%d not posted (%s) -> %s/", kind, number, reason, state)
        return state
    except Exception:
        claim.abort()
        raise


def promote_reviewed(db: filedb.Db, kind: str) -> None:
    """--approve: standing actionable verdicts go out without an operator."""
    for k, number in db.list_state("reviewed"):
        if k == kind and postable(
                ticket_decision(kind, number, db.get("reviewed", kind, number) or {})):
            db.move("reviewed", "outgoing", kind, number)


def send_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
              issue_ns: argparse.Namespace | None, *, dry_run: bool = False) -> None:
    for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
        if ns is None:
            continue
        if getattr(ns, "approve", False):
            promote_reviewed(db, kind)
        outgoing = [n for k, n in db.list_state("outgoing") if k == kind]
        if not outgoing:
            continue
        counts = {action: 0 for action in fairy.ACTIONABLE_DECISIONS}
        cache = gcli_cache.load_cache(ns.cache)
        try:
            for number in outgoing:
                try:
                    send_one(db, ns, kind, number, cache=cache,
                             counts=counts, dry_run=dry_run)
                except Exception:
                    logger.exception("%s #%d: send failed; stays in outgoing/",
                                     kind, number)
        finally:
            gcli_cache.save_cache(ns.cache, cache)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repo agent: scan the forge and maintain the filedb tickets "
                    "for one repository's PRs and issues.",
    )
    p.add_argument("--pr-args", metavar="ARGS",
                   help="the PR side's argument string; ./fairy.py --help "
                        "documents its contents")
    p.add_argument("--issue-args", metavar="ARGS",
                   help="the issue side's argument string; ./issue_fairy.py "
                        "--help documents its contents")
    p.add_argument("--db-root", type=Path,
                   help="filedb root for this repo (default: ~/.fairy/db/<forge~account~owner~repo>)")
    p.add_argument("--loop", type=float, default=0, metavar="SECONDS",
                   help="rescan every N seconds; operator files (requests/, "
                        "outgoing/) wake the loop instantly via watchdog "
                        "(default: one pass, cron style)")
    p.add_argument("--drain", type=int, nargs="?", const=1, default=0,
                   metavar="N",
                   help="run the LLM worker inline between scan and send "
                        "(the whole cycle as one cronjob process), reviewing "
                        "up to N tickets concurrently (bare --drain: 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="log what the send pass would post; post nothing")
    args = p.parse_args(argv)
    if not args.pr_args and not args.issue_args:
        p.error("at least one of --pr-args / --issue-args is required")
    return args


def db_root_for(ns: argparse.Namespace) -> Path:
    return workset.repo_dir(
        Path.home() / ".fairy" / "db",
        forge_type=ns.forge_type, account=ns.gcli_account or "",
        owner=ns.owner, repo=ns.repo)


def one_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
             issue_ns: argparse.Namespace | None,
             args: argparse.Namespace) -> None:
    scan_pass(db, pr_ns, issue_ns)
    if args.drain:
        # Imported here: worker imports agent for db_root_for, so a
        # module-level import back would be circular.
        import worker
        worker.drain(db, {k: v for k, v in (("pr", pr_ns), ("issue", issue_ns))
                          if v is not None}, parallel=args.drain)
    send_pass(db, pr_ns, issue_ns, dry_run=args.dry_run)


def main() -> int:
    args = parse_args()
    pr_ns = fairy.parse_args(shlex.split(args.pr_args)) if args.pr_args else None
    issue_ns = issue_fairy.parse_args(shlex.split(args.issue_args)) if args.issue_args else None
    lead = pr_ns or issue_ns
    setup_logging(fairy.logger, max(ns.verbose for ns in (pr_ns, issue_ns) if ns),
                  logger, workset.logger, gcli_cache.logger, filedb.logger)
    for log_file in {ns.log_file for ns in (pr_ns, issue_ns)
                     if ns and ns.log_file}:
        add_file_log(log_file, fairy.logger, logger, workset.logger,
                     gcli_cache.logger, filedb.logger)
    db = filedb.Db(args.db_root or db_root_for(lead))
    logger.info("agent for %s/%s, db %s", lead.owner, lead.repo, db.root)
    # The forge rescan stays on the --loop interval, but operator files
    # must not wait for it: a request or a y-press (outgoing/) wakes the
    # loop within milliseconds; a wake without a request only needs the
    # send pass, not a full forge scan.
    wake = Event()
    watch_paths([db.root / "requests", db.root / "outgoing"], wake.set)
    next_scan = 0.0
    while True:
        try:
            if time.monotonic() >= next_scan or db.list_state("requests"):
                one_pass(db, pr_ns, issue_ns, args)
                next_scan = time.monotonic() + args.loop
            else:
                send_pass(db, pr_ns, issue_ns, dry_run=args.dry_run)
        except Exception:
            # A transient forge/gcli error must not kill the daemon;
            # one-shot (cron) mode still fails loudly via its exit code.
            if not args.loop:
                raise
            logger.exception("pass failed; retrying in %gs", args.loop)
            next_scan = time.monotonic() + args.loop
        if not args.loop:
            return 0
        wake.wait(max(0.0, next_scan - time.monotonic()))
        wake.clear()


if __name__ == "__main__":
    raise SystemExit(main())
