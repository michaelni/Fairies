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
through fairy/issue_fairy's guarded submit seams. A guard failure
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

import ci_log
import db_config
import fairy
import filedb
import forge_gcli
import gcli_cache
import issue_fairy
import worker
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


def _age_h(data: dict, now: datetime, field: str = "state_changed_at") -> float:
    # iso_to_dt: a hand-edited naive timestamp must degrade to a wrong
    # age, never to a TypeError that kills the scan pass
    changed = iso_to_dt(data.get(field))
    if changed is None:
        return float("inf")
    return (now - changed).total_seconds() / 3600.0


def gate_state(decision: fairy.Decision) -> str:
    """Attention classes get their own directories; the rest is a plain
    gate skip."""
    if decision.action == "error":
        # a failed prepare must be paced and visible, not a silent
        # every-scan refail dressed as a skip
        return "error"
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
        "head_branch": fairy.get_pr_head_branch(item),
        "action": decision.action,
        "reason": decision.reason,
        # the attention rows exist to be acted on in the forge web UI
        "html_url": str(item.get("html_url") or ""),
        # so an operator x (-> cancelled) sticks until the item changes
        "expected_updated_at": item.get("updated_at"),
        "cancelled_ci_contexts": list(decision.cancelled_ci_contexts),
        "blocked_ci_contexts": list(decision.blocked_ci_contexts),
        "external_approvers": list(decision.external_approvers),
        "last_activity_iso": (decision.last_activity.isoformat()
                              if decision.last_activity else None),
        "approved_at": (decision.approved_at.isoformat()
                        if decision.approved_at else None),
    }


def _route(db: filedb.Db, kind: str, number: filedb.TicketId, state: str,
           data: dict, prior: str | None) -> None:
    """Scan-time routing: dst-first, and refused when the item moved at
    all (worker claim, operator y/s/x) during the seconds the prepare
    took -- the scan's decision was made against ``prior`` and is stale
    for anything else."""
    if not db.replace(state, kind, number, data, expect=prior):
        logger.info("%s #%s moved while preparing; not rerouted",
                    kind, number)


def scan_side(db: filedb.Db, ns: argparse.Namespace, kind: str, *,
              now: datetime, cache, self_login,
              forced: set[filedb.TicketId]) -> set[tuple[str, int]]:
    """One gate pass over the side's open items; returns the open set."""
    if kind == "pr":
        fetch_one, list_open, forced_ns = fairy.get_pr, fairy.list_open_prs, \
            ns.force_review_prs
        wip_re = fairy.compile_wip_regex(
            fairy.DEFAULT_WIP_PREFIXES + (ns.wip_prefixes or []))
    else:
        fetch_one, list_open, forced_ns = issue_fairy.get_issue, \
            issue_fairy.list_open_issues, ns.force_review_issues
    # a request may name a sample/review token: the base item is
    # fetched and force-prepared once; tickets are created for each
    # requested evaluation
    evals: dict[int, set] = {}
    forced_base = set()
    for token in forced:
        base = filedb.forge_number(token)
        forced_base.add(base)
        if not filedb.is_base(token):
            evals.setdefault(base, set()).add(token)
    # a samples-only request force-prepares the base as the payload
    # template but must not queue a base evaluation nobody asked for
    template_only = {b for b in evals
                     if str(b) not in forced and b not in forced_ns}
    if ns.forced_only:  # --forced-only: no open listing, just the named items
        items = []
        missing = sorted(forced_ns | forced_base)
    else:
        items = list_open(ns)
        listed = {int(i["number"]) for i in items
                  if str(i.get("number")).isdigit()}
        # forced/requested numbers may be closed or merged: absent from
        # the open listing but explicitly asked for
        missing = sorted((forced_ns | forced_base) - listed)
    for n in missing:
        try:
            items.append(fetch_one(ns, n))
        except Exception as exc:
            logger.error("%s #%s: forced fetch failed: %s", kind, n, exc)
            # an error ticket marks the request consumed and puts the
            # failure on screen; an existing ticket already does both
            # (find() reporting the request file itself counts as none)
            if db.find(kind, str(n)) in (None, "requests"):
                db.push("error", kind, str(n),
                        {"error": f"forced fetch failed: {exc}"})
    # Request-forced numbers bypass the gates through the same ns set
    # the gates read; the addition is undone after the pass so a
    # request does not force every future scan.
    added_forced = forced_base - forced_ns
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
                    cache_age=cache_age, limit=limit, evals=evals,
                    template_only=template_only)
    finally:
        forced_ns -= added_forced
    return open_set


def _scan_items(db, ns, kind, items, *, now, cache, self_login, forced_ns,
                wip_re, cache_age, limit, evals={},
                template_only=frozenset()) -> None:
    queued = 0
    for item in sorted(items, key=lambda i: (int(i["number"]) not in forced_ns,
                                             int(i["number"]))):
        number = int(item["number"])
        token = str(number)
        prior = db.find(kind, token)
        if prior in IN_FLIGHT:
            continue
        prior_data = db.get(prior, kind, token) if prior else None
        if prior == "reviewed" and number not in forced_ns and prior_data:
            # Any guard-matching verdict stands -- including skips that
            # carry label changes: they sit in reviewed/ awaiting the
            # operator, and a re-queue would burn an LLM run and yank
            # the row out from under the cursor every scan.
            if (prior_data.get("expected_updated_at") == item.get("updated_at")
                    and (kind != "pr" or prior_data.get("expected_head_ref")
                         == fairy.get_pr_head_ref(item))):
                continue  # standing verdict; reuse
            if prior_data.get("send_blocked") and not getattr(ns, "approve", False):
                # A blocked y in manual mode is the operator's case:
                # requeueing here replaced their verdict with whatever
                # the fresh round produced -- a gate skip destroyed an
                # approve (production: #23863, genuine data loss). They
                # answer with r, s or Y; only auto mode re-reviews.
                continue
        if prior == "cancelled" and number not in forced_ns and prior_data \
                and str(prior_data.get("reason", "")).startswith("operator") \
                and prior_data.get("expected_updated_at") == item.get("updated_at"):
            # the operator threw it out; only new activity revives it.
            # Closure cancels ("not open") deliberately do NOT stick: a
            # transiently short forge listing must cost one redundant
            # review at most, never a permanently dead verdict.
            continue
        backoff_h = 0.0
        in_backoff_window = False
        if prior == "skipped" and prior_data and (
                prior_data.get("llm_at") or prior_data.get("snoozed_at")):
            # An LLM skip serves its doubling backoff in skipped/; the
            # file is the memory, so it must not be refreshed early.
            # New activity bypasses the wait outright: a push, comment
            # or @-mention must reach the gates now, not in days. An
            # unchanged updated_at means nothing at all changed, so the
            # wait is served without any further fetch; a moved
            # updated_at is judged after prepare (below), because only
            # the discussion tells label edits apart from real activity.
            wait = backoff_wait_h(prior_data.get("skip_backoff_h", 0))
            # the window is measured from the LLM run or the operator's
            # s-press, whichever is later -- so skipping an old verdict
            # is a real snooze, and the label-edit refresh below (which
            # re-stamps state_changed_at) cannot extend anything
            age = min(_age_h(prior_data, now, "llm_at"),
                      _age_h(prior_data, now, "snoozed_at"))
            served = number not in forced_ns and age < wait
            if served and prior_data.get("expected_updated_at") == item.get("updated_at"):
                continue
            in_backoff_window = served
            # a changed item re-enters without doubling: the doubling
            # counts served waits, not bypasses
            backoff_h = float(prior_data.get("skip_backoff_h") or 0) \
                if served else wait
        elif prior == "skipped" and prior_data:
            # a timestamp-less skip (operator s, send guard-fail) is
            # re-gated now but keeps the LLM-skip escalation, which it
            # says nothing about
            backoff_h = float(prior_data.get("skip_backoff_h") or 0)
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
                # an error/ ticket gives the failure a row, a summary
                # line and ERROR_RETRY_H pacing; the archive and a
                # standing verdict outrank it
                if prior not in ("posted", "cancelled", "reviewed"):
                    _route(db, kind, token, "error", {
                        "title": str(item.get("title") or ""),
                        "html_url": str(item.get("html_url") or ""),
                        "error": f"prepare failed: {exc}",
                        "expected_updated_at": item.get("updated_at"),
                    }, prior)
                continue
        if isinstance(prepared, fairy.Decision):
            state = gate_state(prepared)
            # A plain gate skip must not clobber the archive: posted/
            # cancelled/error records outrank "nothing to do today".
            if state == "skipped" and prior in ("posted", "cancelled", "error",
                                                "skipped"):
                if prior != "skipped" or (prior_data or {}).get("llm_at"):
                    continue
            ticket = gate_ticket(prepared, item)
            if state == "error":
                if prior in ("posted", "cancelled", "reviewed"):
                    continue  # never clobber archive or a standing verdict
                ticket["error"] = prepared.reason
            _route(db, kind, token, state, ticket, prior)
            for token in evals.get(number, ()):
                # a requested evaluation of a gate-skipped item cannot
                # be built (no payload); the error ticket consumes the
                # request and puts the reason on screen
                if db.find(kind, token) in (None, "requests"):
                    db.push("error", kind, token,
                            {"error": f"base item gate-skipped: "
                                      f"{prepared.reason}"})
            continue
        if in_backoff_window:
            # updated_at moved during the wait, but only real activity
            # bypasses it: a label/milestone edit bumps updated_at
            # without touching head or discussion, and must not burn an
            # LLM run early -- the bypass keys on the head+last_activity
            # pair, deliberately not on updated_at.
            last_iso = (prepared.last_activity.isoformat()
                        if prepared.last_activity else None)
            if last_iso == prior_data.get("last_activity_iso") \
                    and (kind != "pr" or prior_data.get("expected_head_ref")
                         == fairy.get_pr_head_ref(item)):
                db.try_move("skipped", "skipped", kind, token,
                            mutate=lambda d: d.update(
                                expected_updated_at=item.get("updated_at")))
                continue
        if limit and queued >= limit and number not in forced_ns \
                and not getattr(prepared, "forced_review", False):
            # nothing written: --limit never persists a skip -- but say
            # so, or the deferral is invisible everywhere
            logger.info("%s #%d eligible but over --limit; next pass retries",
                        kind, number)
            continue
        ticket = {
            "title": prepared.title,
            "author": prepared.author,
            "head_branch": fairy.get_pr_head_branch(item),
            # queued/error tickets need the guard too, or an operator x
            # on them cannot stick until new activity
            "expected_updated_at": item.get("updated_at"),
            "html_url": str(getattr(prepared, "pr", getattr(prepared, "issue", {})).get("html_url") or ""),
            "skip_backoff_h": backoff_h,
            "forced": number in forced_ns,
            "prepared": fairy.prepared_to_dict(prepared),
        }
        if number not in template_only:
            _route(db, kind, token, "queued", ticket, prior)
        for token in evals.get(number, ()):
            # one prepare, one payload copy per requested evaluation;
            # samples never re-enter via gates/backoff (scan keys on
            # forge numbers only), so a skipped sample is final
            _route(db, kind, token, "queued", dict(ticket),
                   db.find(kind, token))
        queued += 1
        logger.info("%s #%d queued (backoff %gh, %d/%s)", kind, number,
                    backoff_h, queued, limit or "inf")


def consume_requests(db: filedb.Db) -> dict[str, set[filedb.TicketId]]:
    """Requests force a fresh gate-bypassing ticket; they are deleted
    only after the ticket exists (at-least-once)."""
    forced: dict[str, set[filedb.TicketId]] = {"pr": set(), "issue": set()}
    for kind, number in db.list_state("requests"):
        forced[kind].add(number)
    return forced


def finish_requests(db: filedb.Db, forced: dict[str, set[filedb.TicketId]],
                    kinds: set[str]) -> None:
    """Drop the requests this pass consumed, but only once some ticket
    exists for the item (at-least-once: a crashed pass retries). A
    request that arrived mid-pass is not in ``forced`` and waits; a
    request for a kind this agent never scanned is not ours to consume
    (a pr-only agent must not eat an issue rerun)."""
    for kind, numbers in forced.items():
        if kind not in kinds:
            continue
        for number in numbers:
            # find() would report the request file itself; try_pop so a
            # worker's held review lock can never stall the agent pass
            if db.find(kind, number) not in (None, "requests"):
                db.try_pop("requests", kind, number)


def closure_reason(ns: argparse.Namespace, kind: str, number) -> str | None:
    """One fetch to name WHY an item left the open listing: "merged" is
    the success story and must not read as a failure in the UI
    (production: #23913 showed plain cancelled after the operator
    merged it). None means the item is in fact still open -- the
    listing was transiently short -- and must not be cancelled at all.
    A failed fetch keeps the old revivable "not open"."""
    try:
        if kind == "pr":
            item = fairy.get_pr(ns, filedb.forge_number(number))
        else:
            item = issue_fairy.get_issue(ns, filedb.forge_number(number))
    except Exception as exc:
        logger.warning("%s #%s left the listing but the fate fetch "
                       "failed: %s", kind, number, exc)
        return "not open"
    if kind == "pr" and item.get("merged"):
        return "merged"
    if item.get("state") == "closed":
        return "closed without merge" if kind == "pr" else "closed"
    return None


def cancel_closed(db: filedb.Db, open_set: set[tuple[str, int]],
                  kinds: set[str],
                  nss: dict[str, argparse.Namespace]) -> None:
    # attention tickets too: a merged PR's merge-ready/ci-blocked row
    # would otherwise sit there forever (prune skips non-settled states)
    for state in ("queued", "reviewed", "ci-blocked", "merge-ready",
                  "awaiting-approver"):
        for kind, number in db.list_state(state):
            if kind in kinds \
                    and (kind, filedb.forge_number(number)) not in open_set:
                reason = closure_reason(nss[kind], kind, number)
                if reason is None:
                    continue
                # try_move: a claimed item's lock is held for the whole
                # review and must not stall the pass; retried next scan
                if db.try_move(state, "cancelled", kind, number,
                               mutate=lambda d, r=reason: d.update(reason=r)):
                    logger.info("%s #%s cancelled: %s", kind, number, reason)


def scan_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
              issue_ns: argparse.Namespace | None,
              now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    forced = consume_requests(db)
    open_set: set[tuple[str, int]] = set()
    kinds: set[str] = set()
    # A --forced-only side's open_set is just the named items, not the
    # open listing: everything else would look closed, so closing and
    # pruning are skipped for it.
    full_kinds: set[str] = set()
    for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
        if ns is None:
            continue
        kinds.add(kind)
        if not ns.forced_only:
            full_kinds.add(kind)
        cache = gcli_cache.load_cache(ns.cache)
        try:
            self_login = forge_gcli.self_login(ns)
            open_set |= scan_side(db, ns, kind, now=now, cache=cache,
                                  self_login=self_login, forced=forced[kind])
        finally:
            gcli_cache.save_cache(ns.cache, cache)
    finish_requests(db, forced, kinds)
    cancel_closed(db, open_set, full_kinds,
                  {"pr": pr_ns, "issue": issue_ns})
    for kind, number in db.reap():
        logger.warning("%s #%s re-queued: its worker died", kind, number)
    # A crash between a transition's dst-write and src-unlink leaves the
    # item in two states, and the shadowed file stays live bait: a
    # reviewed/ one behind outgoing/ re-posts the verdict, a queued/ one
    # behind an archive state buys an LLM run whose verdict find() then
    # hides. requests/ is a command channel that coexists by design;
    # llm/ was just reaped with its re-queue.
    for state in filedb.STATES:
        if state not in ("requests", "llm"):
            db.reap(state, None)
    if full_kinds == kinds:  # prune's keep-set is kind-blind
        for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
            if ns is None:
                continue
            before = now - timedelta(days=ns.workset_retention_days)
            for state in ("posted", "skipped", "cancelled", "error"):
                db.prune(state, before, keep=open_set, kinds={kind},
                         key=lambda kn: (kn[0], filedb.forge_number(kn[1])))


def ticket_decision(kind: str, number, ticket: dict) -> fairy.Decision | None:
    """Rebuild a postable Decision purely from a verdict ticket; the
    ticket's guard rides on the Decision so the staleness checks pin
    the post to the reviewed state."""
    review = ticket.get("review") or {}
    if not review.get("classification"):
        return None
    llm = fairy.LLMReview(
        classification=review["classification"],
        message=review.get("message", ""),
        # field-by-field, not **c: tickets are hand-editable and one
        # typo'd key must not kill the TUI or wedge the agent loop
        label_changes=tuple(
            fairy.LabelChange(str(c.get("label") or ""), str(c.get("op") or ""),
                              str(c.get("reason") or ""), bool(c.get("post")))
            for c in review.get("label_changes") or () if isinstance(c, dict)))
    if kind == "pr":
        decision = fairy.decision_from_review(
            llm, number=filedb.forge_number(number),
            title=ticket.get("title", ""),
            author=ticket.get("author", ""),
            auto_merge=ticket.get("auto_merge", "-"),
            last_activity=iso_to_dt(ticket.get("last_activity_iso")),
            base_reason="persisted review")
    else:
        decision = issue_fairy.issue_review_decision(
            llm, number=filedb.forge_number(number),
            title=ticket.get("title", ""),
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
                  *, cache, counts: dict[str, int],
                  skip_guard: bool = False) -> str | None:
    """The forge side effects, through fairy/issue_fairy's guarded
    submit seams; the staleness guard's block reason when it refused,
    None when everything went out. ``skip_guard`` is the operator's
    force_post riding through to the seams."""
    if kind == "issue":
        return issue_fairy.submit_issue_decision(
            ns, decision, cache=cache, submitted_counts=counts,
            skip_guard=skip_guard)
    if decision.action in fairy.ACTIONABLE_DECISIONS:
        reason = fairy.submit_decision_action(ns, decision, decision, cache=cache,
                                              skip_guard=skip_guard)
        if reason is not None:
            return reason
        counts[decision.action] += 1
        if fairy.decision_has_label_changes(decision):
            fairy.apply_triage_labels(ns, decision, decision, skip_guard=True)
        return None
    return fairy.apply_triage_labels(ns, decision, decision, skip_guard=skip_guard)


def send_one(db: filedb.Db, ns: argparse.Namespace, kind: str,
             number: filedb.TicketId, *,
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
            logger.info("%s #%s: DRY RUN, would post: %s", kind, number,
                        fairy.manual_action_description(decision))
            claim.abort()
            return None
        # the operator's Y: post as-is although the item may have
        # moved since the review; popped so the archive stays clean
        reason = post_decision(ns, kind, decision, cache=cache, counts=counts,
                               skip_guard=bool(ticket.pop("force_post", None)))
        if reason is None:
            claim.finish("posted", dict(
                ticket, posted_at=datetime.now(timezone.utc).isoformat()))
            logger.info("%s #%s posted: %s", kind, number,
                        fairy.manual_action_description(decision))
            return "posted"
        if getattr(ns, "approve", False):
            # Auto mode must not stall on a stale verdict: without
            # llm_at the skipped/ ticket is re-gated (and, the item
            # having changed, freshly re-reviewed) on the next scan.
            # skip_backoff_h is kept: the guard failing says nothing
            # about the item's earned skip history.
            ticket.pop("llm_at", None)
            state = "skipped"
        else:
            state = "reviewed"
        claim.finish(state, dict(ticket, send_blocked=reason))
        logger.info("%s #%s not posted (%s) -> %s/", kind, number, reason, state)
        return state
    except Exception:
        claim.abort()
        raise


def promote_reviewed(db: filedb.Db, kind: str) -> None:
    """--approve: standing actionable verdicts go out without an operator."""
    for k, number in db.list_state("reviewed"):
        # find() precedence: a reviewed/ crash remnant behind a later
        # state must not be promoted (and posted) a second time
        if k == kind and filedb.is_base(number) \
                and db.find(kind, number) == "reviewed" and postable(
                ticket_decision(kind, number, db.get("reviewed", kind, number) or {})):
            db.try_move("reviewed", "outgoing", kind, number)


def log_summary(db: filedb.Db) -> None:
    """End-of-pass report from the directories: which
    verdicts an operator could apply right now, and which items need a
    human's CI/approval action -- with the details, not just counts."""
    ready = []
    for kind, number in db.list_state("reviewed"):
        t = db.get("reviewed", kind, number) or {}
        if t.get("action") in fairy.ACTIONABLE_DECISIONS:
            ready.append((kind, number, t.get("action"), t.get("title", "")))
    if ready:
        logger.info("reviewed/ awaiting you: %d", len(ready))
        for kind, number, action, title in ready:
            logger.info("  %s #%-6s %-15s %s", kind, number, action, title)
    for state, label in (("merge-ready", "approved, ready to apply"),
                         ("ci-blocked", "CI needs a human"),
                         ("awaiting-approver", "waiting for an approver")):
        items = db.list_state(state)
        if not items:
            continue
        logger.info("%s/ (%s): %d", state, label, len(items))
        for kind, number in items:
            t = db.get(state, kind, number) or {}
            detail = ", ".join((t.get("cancelled_ci_contexts") or [])
                               + (t.get("blocked_ci_contexts") or [])
                               + (t.get("external_approvers") or []))
            logger.info("  %s #%-6s %s%s", kind, number, t.get("title", ""),
                        f"  [{detail}]" if detail else "")


def ask_pass(db: filedb.Db, kinds: set[str], retry=None) -> None:
    """--ask: the pre-TUI prompt flow. Print each actionable reviewed/
    verdict (URL, action, message) and ask; y hands it to the send
    pass via outgoing/, s skips one-shot (the next scan reconsiders),
    S snoozes (>=24h doubling), x cancels, l(ater)/enter leaves it, r
    requeues it for a fresh review (``retry`` produces it inline and
    the fresh verdict is asked again; without an inline worker the
    request waits for one), q stops asking. Rows a worker holds are
    simply skipped this round."""
    for kind, number in db.list_state("reviewed"):
        if kind not in kinds or not filedb.is_base(number):
            continue
        asking = True
        while asking and db.find(kind, number) == "reviewed":
            asking = False
            ticket = db.get("reviewed", kind, number) or {}
            decision = ticket_decision(kind, number, ticket)
            if not postable(decision):
                break
            print(f"\n{ticket.get('html_url') or f'{kind} #{number}'}"
                  f"  {ticket.get('title', '')}")
            print(fairy.manual_action_description(decision))
            message = (ticket.get("review") or {}).get("message") or ""
            if message:
                print(message)
            while True:
                try:
                    choice = input(f"{kind} #{number}: post? [y]es/[s]kip/"
                                   "[S]nooze/[x] cancel/[r]etry/[l]ater/[q]uit ")
                except EOFError:
                    return
                raw = choice.strip()
                choice = raw.lower()
                if choice in ("y", "yes"):
                    db.try_move("reviewed", "outgoing", kind, number)
                    break
                if raw == "S" or choice == "snooze":
                    db.try_move("reviewed", "skipped", kind, number,
                                mutate=lambda d: d.update(
                                    reason="operator snooze",
                                    snoozed_at=datetime.now(timezone.utc).isoformat()))
                    break
                if raw == "s" or choice == "skip":

                    def skip_now(d: dict) -> None:
                        d["reason"] = "operator skip"
                        d.pop("llm_at", None)
                        d.pop("snoozed_at", None)

                    db.try_move("reviewed", "skipped", kind, number,
                                mutate=skip_now)
                    break
                if choice in ("x", "cancel"):
                    db.try_move("reviewed", "cancelled", kind, number,
                                mutate=lambda d: d.update(reason="operator cancel"))
                    break
                if choice in ("r", "retry"):
                    db.push("requests", kind, number, {"action": "rerun"})
                    if retry is None:
                        print("rerun requested; a worker will pick it up")
                    else:
                        retry()
                        asking = True
                        if db.find(kind, number) != "reviewed":
                            print(f"{kind} #{number}: the rerun verdict landed "
                                  f"in {db.find(kind, number)}/")
                    break
                if choice in ("", "l", "later"):
                    break
                if choice in ("q", "quit"):
                    return
                print("please answer y, s, S, x, r, l or q")


def send_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
              issue_ns: argparse.Namespace | None, *, dry_run: bool = False) -> None:
    for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
        if ns is None:
            continue
        if getattr(ns, "approve", False):
            if dry_run:
                # promotion is a persistent staging step: a later normal
                # run would post whatever a dry preview promoted
                for k, number in db.list_state("reviewed"):
                    if k == kind and postable(ticket_decision(
                            k, number, db.get("reviewed", k, number) or {})):
                        logger.info("%s #%s: DRY RUN, would promote to "
                                    "outgoing/", k, number)
            else:
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
                    logger.exception("%s #%s: send failed; stays in outgoing/",
                                     kind, number)
        finally:
            gcli_cache.save_cache(ns.cache, cache)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repo agent: scan the forge and maintain the filedb tickets "
                    "for one repository's PRs and issues.",
    )
    p.add_argument("--db-root", type=Path, required=True,
                   help="filedb root; its config.toml, written by "
                        "configurator.py, carries the side argument strings")
    p.add_argument("--loop", type=float, default=0, metavar="SECONDS",
                   help="rescan every N seconds; operator files (requests/, "
                        "outgoing/) wake the loop instantly via watchdog "
                        "(default: one pass, cron style)")
    p.add_argument("--drain", type=int, nargs="?", const=1, default=0,
                   metavar="N",
                   help="run the LLM worker inline between scan and send "
                        "(the whole cycle as one cronjob process), reviewing "
                        "up to N tickets concurrently (bare --drain: 1)")
    p.add_argument("--ask", action="store_true",
                   help="prompt per reviewed verdict before the send pass "
                        "(the pre-TUI manual flow: y posts, s skips one-shot, "
                        "S snoozes, x cancels, r reruns the review, l defers, "
                        "q stops)")
    p.add_argument("--dry-run", action="store_true",
                   help="log what the send pass would post; post nothing")
    return p.parse_args(argv)


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
        "    head pinned at --patch-pr-ref-template. Fairy trusts those\n"
        "    refs verbatim; nothing here verifies they match the cutoff.\n"
        "  * Pass --cache and --db-root <separate paths> to keep the live\n"
        "    caches and filedb clean.",
        ignore_after.isoformat(),
    )


def one_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
             issue_ns: argparse.Namespace | None,
             args: argparse.Namespace) -> None:
    sides = {k: v for k, v in (("pr", pr_ns), ("issue", issue_ns))
             if v is not None}

    def review_cycle() -> None:
        scan_pass(db, pr_ns, issue_ns)
        if args.drain:
            worker.drain(db, sides, parallel=args.drain)

    review_cycle()
    if args.ask:
        ask_pass(db, set(sides),
                 retry=review_cycle if args.drain else None)
    send_pass(db, pr_ns, issue_ns, dry_run=args.dry_run)
    log_summary(db)


def _forced_only(ns: argparse.Namespace | None) -> argparse.Namespace | None:
    if ns is None:
        return None
    clone = argparse.Namespace(**vars(ns))
    clone.forced_only = True
    return clone


def requests_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
                  issue_ns: argparse.Namespace | None,
                  args: argparse.Namespace) -> None:
    """Answer pending operator requests by fetching just the named
    items (the --forced-only path). A full forge rescan per r/f press
    is hammering, and it kept the press invisible for the length of a
    scan (production: ~74s for one issue); the scheduled full scan
    stays on its own clock."""
    one_pass(db, _forced_only(pr_ns), _forced_only(issue_ns), args)


def main() -> int:
    args = parse_args()
    pr_args, issue_args = db_config.read_side_strings(args.db_root)
    pr_ns = fairy.parse_args(shlex.split(pr_args)) if pr_args else None
    issue_ns = issue_fairy.parse_args(shlex.split(issue_args)) if issue_args else None
    lead = pr_ns or issue_ns
    setup_logging(fairy.logger, max(ns.verbose for ns in (pr_ns, issue_ns) if ns),
                  logger, db_config.logger, workset.logger, gcli_cache.logger,
                  filedb.logger, forge_gcli.logger, ci_log.logger,
                  color=lead.color)
    log_files = {ns.log_file for ns in (pr_ns, issue_ns) if ns and ns.log_file}
    for log_file in log_files:
        add_file_log(log_file, fairy.logger, logger, db_config.logger,
                     workset.logger, gcli_cache.logger, filedb.logger,
                     forge_gcli.logger, ci_log.logger)
    db = filedb.Db(args.db_root)
    logger.info("agent for %s/%s, db %s", lead.owner, lead.repo, db.root)
    for kind, args_str in (("pr", pr_args), ("issue", issue_args)):
        if args_str:
            logger.info("%s side from %s: %s", kind, db_config.CONFIG_NAME,
                        args_str)
    for ns in (pr_ns, issue_ns):
        if ns is not None and getattr(ns, "simulate_past", None):
            warn_simulate_past_limitations(ns.simulate_past)
    # The forge rescan stays on the --loop interval, but operator files
    # must not wait for it: a request or a y-press (outgoing/) wakes the
    # loop within milliseconds; a wake without a request only needs the
    # send pass, not a full forge scan.
    wake = Event()
    watch_paths([db.root / "requests", db.root / "outgoing"], wake.set)
    next_scan = 0.0
    while True:
        try:
            if time.monotonic() >= next_scan:
                one_pass(db, pr_ns, issue_ns, args)
                next_scan = time.monotonic() + args.loop
            elif db.list_state("requests"):
                requests_pass(db, pr_ns, issue_ns, args)
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
