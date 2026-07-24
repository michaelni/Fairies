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

What belongs here: the scan pass and the ticket routing policy.
What does NOT belong: file atomicity (filedb), gates and payload
building (fairy / issue_fairy), LLM work (the worker), the UI
(fairy_tui).
"""

from __future__ import annotations

import argparse
import logging
import shlex
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fairy
import filedb
import gcli_cache
import issue_fairy
import workset
from common import default_cache_path, setup_logging

__all__ = ["main", "scan_pass"]

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


def gate_ticket(decision: fairy.Decision) -> dict:
    return {
        "title": decision.title,
        "author": decision.author,
        "action": decision.action,
        "reason": decision.reason,
        "cancelled_ci_contexts": list(decision.cancelled_ci_contexts),
        "blocked_ci_contexts": list(decision.blocked_ci_contexts),
        "external_approvers": list(decision.external_approvers),
    }


def _route(db: filedb.Db, kind: str, number: int, state: str, data: dict) -> None:
    prior = db.find(kind, number)
    if prior is not None and prior != state:
        db.pop(prior, kind, number)
    db.push(state, kind, number, data)


def scan_side(db: filedb.Db, ns: argparse.Namespace, kind: str, *,
              now: datetime, cache, self_login, forced: set[int]) -> set[tuple[str, int]]:
    """One gate pass over the side's open items; returns the open set."""
    if kind == "pr":
        items = fairy.list_open_prs(ns)
        wip_re = fairy.compile_wip_regex(
            fairy.DEFAULT_WIP_PREFIXES + (ns.wip_prefixes or []))
        forced_ns = ns.force_review_prs
    else:
        items = issue_fairy.list_open_issues(ns)
        forced_ns = ns.force_review_issues
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
            verdict = (prior_data.get("review") or {}).get("classification")
            if (verdict not in ("skip", "error", "-", "", None)
                    and prior_data.get("expected_updated_at") == item.get("updated_at")
                    and (kind != "pr" or prior_data.get("expected_head_ref")
                         == fairy.get_pr_head_ref(item))):
                continue  # standing verdict; reuse
        backoff_h = 0.0
        if prior == "skipped" and prior_data and prior_data.get("llm_at"):
            # An LLM skip serves its doubling backoff in skipped/; the
            # file is the memory, so it must not be refreshed early.
            wait = backoff_wait_h(prior_data.get("skip_backoff_h", 0))
            if number not in forced_ns and _age_h(prior_data, now) < wait:
                continue
            backoff_h = wait
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
            _route(db, kind, number, state, gate_ticket(prepared))
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


def finish_requests(db: filedb.Db) -> None:
    for kind, number in db.list_state("requests"):
        if db.find(kind, number) in ("queued", "llm", "reviewed"):
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
    for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
        if ns is None:
            continue
        kinds.add(kind)
        cache = gcli_cache.load_cache(ns.cache)
        try:
            self_login = fairy.get_self_login(ns)
            open_set |= scan_side(db, ns, kind, now=now, cache=cache,
                                  self_login=self_login, forced=forced[kind])
        finally:
            gcli_cache.save_cache(ns.cache, cache)
    finish_requests(db)
    cancel_closed(db, open_set, kinds)
    for kind, number in db.reap():
        logger.warning("%s #%d re-queued: its worker died", kind, number)
    retention_ns = pr_ns or issue_ns
    before = now - timedelta(days=retention_ns.workset_retention_days)
    for state in ("posted", "skipped", "cancelled", "error"):
        db.prune(state, before, keep=open_set)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repo agent: scan the forge and maintain the filedb tickets "
                    "for one repository's PRs and issues.",
    )
    p.add_argument("--pr-args", metavar="ARGS",
                   help="fairy.py argument string for the PR side")
    p.add_argument("--issue-args", metavar="ARGS",
                   help="issue_fairy.py argument string for the issue side")
    p.add_argument("--db-root", type=Path,
                   help="filedb root for this repo (default: ~/.fairy/db/<forge~account~owner~repo>)")
    p.add_argument("--loop", type=float, default=0, metavar="SECONDS",
                   help="rescan every N seconds (default: one pass, cron style)")
    args = p.parse_args(argv)
    if not args.pr_args and not args.issue_args:
        p.error("at least one of --pr-args / --issue-args is required")
    return args


def db_root_for(ns: argparse.Namespace) -> Path:
    return workset.repo_dir(
        Path.home() / ".fairy" / "db",
        forge_type=ns.forge_type, account=ns.gcli_account or "",
        owner=ns.owner, repo=ns.repo)


def main() -> int:
    args = parse_args()
    pr_ns = fairy.parse_args(shlex.split(args.pr_args)) if args.pr_args else None
    issue_ns = issue_fairy.parse_args(shlex.split(args.issue_args)) if args.issue_args else None
    lead = pr_ns or issue_ns
    setup_logging(fairy.logger, max(ns.verbose for ns in (pr_ns, issue_ns) if ns),
                  logger, workset.logger, gcli_cache.logger)
    db = filedb.Db(args.db_root or db_root_for(lead))
    logger.info("agent for %s/%s, db %s", lead.owner, lead.repo, db.root)
    while True:
        started = time.monotonic()
        scan_pass(db, pr_ns, issue_ns)
        if not args.loop:
            return 0
        time.sleep(max(0.0, args.loop - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
