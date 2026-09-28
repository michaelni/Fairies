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
backoff in the ticket. A reviewed/ verdict stands until the operator
moves it; only auto mode requeues one whose item changed (guard
mismatch). Operator requests (requests/) are
agent-mediated so the UI never needs forge access; they bypass gates
and the limit. The agent also reaps dead workers' claims, cancels
tickets whose item left the open listing, and prunes settled tickets.

A send pass follows each scan: every outgoing/ item is re-read under
its claim lock, guard-checked against the live forge and posted
through fairy/issue_fairy's guarded submit seams. A guard failure
returns the verdict to reviewed/ with a note or, under --auto-mode for
a verdict the send pass promoted itself, re-gates it via skipped/ so a
cron run never stalls. --dry-run logs what would be posted and posts
nothing.

What belongs here: the scan pass, the ticket routing policy and the
send pass.
What does NOT belong: file atomicity (filedb), gates and payload
building (fairy / issue_fairy), LLM work (the worker), the UI
(fairy_tui).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from threading import Event
from collections.abc import Callable
from dataclasses import replace as dataclasses_replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import ci_log
import db_config
import fairy
import filedb
import forge_gcli
import halt_marker
import gcli_cache
import git_util
import issue_fairy
import worker
import workset
from forgejo_export import labels
from common import (OVERRIDE_EPILOG, add_file_log, add_grouped_help,
                    config_option_groups, grouped_help,
                    iso_to_dt, options_argv, parse_scoped_overrides,
                    side_actions, setup_logging, split_side_actions,
                    WATCH_FALLBACK_POLL_S, watch_paths)

__all__ = ["main", "scan_pass", "send_pass",
           "REASON_MERGED", "REASON_CLOSED_UNMERGED", "REASON_CLOSED"]

logger = logging.getLogger(__name__)

MIN_BACKOFF_H = 24.0
# States the agent never touches during a scan: the item is being
# worked on or awaits the operator/sender.
IN_FLIGHT = ("queued", "llm", "outgoing")


class Halted(BaseException):
    """The scan sighted --halt-keyword and halted the db. Not an
    Exception: the scan's per-item catch-alls must let it through."""


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
        "body": str(item.get("body") or ""),
        "head_branch": fairy.get_pr_head_branch(item),
        "action": decision.action,
        "reason": decision.reason,
        # the attention rows exist to be acted on in the forge web UI
        "html_url": str(item.get("html_url") or ""),
        # so an operator x (-> cancelled) sticks until the item changes
        "expected_updated_at": item.get("updated_at"),
        "cancelled_ci_contexts": list(decision.cancelled_ci_contexts),
        "blocked_ci_contexts": list(decision.blocked_ci_contexts),
        "ci_pending": decision.ci_pending,
        "external_approvers": list(decision.external_approvers),
        "last_activity_iso": (decision.last_activity.isoformat()
                              if decision.last_activity else None),
        "approved_at": (decision.approved_at.isoformat()
                        if decision.approved_at else None),
    }


def _fetch_thread(ns: argparse.Namespace, kind: str, item: dict, cache,
                  cache_age: timedelta) -> tuple[list, list, list, list]:
    """The item's (reviews, comments, review comments, timeline) --
    ``build_llm_discussion``'s argument order -- through the same cache
    the gates read; fetched first in the pass, so a prepare of the same
    item hits the cache. Issues have no reviews or review comments."""
    if kind == "pr":
        return fairy.get_pr_thread(ns, item, cache=cache,
                                   cache_max_age=cache_age)
    comments, timeline = issue_fairy.get_issue_discussion(
        ns, item, cache=cache, cache_max_age=cache_age)
    return [], comments, [], timeline


def _fetch_pr_head(ns: argparse.Namespace, number: int,
                   head_sha: str | None) -> None:
    """Try one fetch of the forge remote of --patch-repo when its
    ``<remote>/pr/<n>`` ref is not at ``head_sha``, the head the forge
    reports. The fetch can fail or come back without the head; either
    is logged, the snapshot is written regardless, and the ref is
    checked again on the next scan. Not under --simulate-past, whose
    mirror carries operator-pinned refs."""
    repo = ns.patch_repo
    if repo is None or ns.simulate_past or head_sha is None:
        return
    refs = [f"{remote}/pr/{number}" for remote in git_util.FORGE_REMOTES]
    if git_util.git_resolve_first(repo, refs) == head_sha:
        return
    started = time.monotonic()
    try:
        remote = git_util.git_forge_remote(repo)
        logger.info("pr #%s: head %s is not at %s/pr/%s: git -C %s fetch %s",
                    number, head_sha[:12], remote, number, repo, remote)
        git_util.git_fetch(repo, remote, timeout_s=120.0)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        logger.warning("pr #%s: head %s not fetched: %s", number,
                       head_sha[:12], exc)
        return
    logger.info("pr #%s: fetched %s in %.1fs; %s/pr/%s is now %s", number,
                remote, time.monotonic() - started, remote, number,
                (git_util.git_resolve_first(repo, refs) or "absent")[:12])


def _patch_ids(db: filedb.Db, ns: argparse.Namespace, kind: str,
               token: filedb.TicketId, base_sha: str | None,
               head_sha: str | None) -> list[str] | None:
    """The patch-ids of the PR's base..head series in --patch-repo,
    kept from the standing snapshot while base and head are unchanged;
    None without a mirror, under --simulate-past, and while the mirror
    lacks the head."""
    if ns.patch_repo is None or ns.simulate_past or None in (base_sha, head_sha):
        return None
    stored = db.get(filedb.ITEM_STATE, kind, token) or {}
    if (stored.get("base_sha"), stored.get("head_sha")) == (base_sha, head_sha) \
            and stored.get("patch_ids") is not None:
        return stored["patch_ids"]
    logger.debug("%s #%s: patch-ids of %s..%s in %s", kind, token,
                 base_sha[:12], head_sha[:12], ns.patch_repo)
    try:
        return git_util.git_series_patch_ids(ns.patch_repo, base_sha, head_sha)
    except RuntimeError as exc:
        logger.warning("%s #%s: patch-ids of %s..%s not computed: %s",
                       kind, token, base_sha[:12], head_sha[:12], exc)
        return None


def _put_snapshot(db: filedb.Db, ns: argparse.Namespace, kind: str,
                  token: filedb.TicketId, item: dict, cache,
                  cache_age: timedelta) -> dict | None:
    """Refresh the item's filedb snapshot: its fields, discussion and
    forge status -- for a PR the "open"/"closed"/"merged" state, the
    auto-merge schedule, the approval and change-request counts
    (latest non-stale review per author) and the series' patch-ids,
    for an issue the "open"/"closed" state and its labels. A failure
    costs freshness, never the scan of the item; returns the snapshot
    written, None when it could not be refreshed. Raises Halted, after
    halting the db, when the item's body, a comment, a review or a
    review comment carries --halt-keyword; the reason names the poster
    and links the post."""
    try:
        reviews, comments, review_comments, timeline = _fetch_thread(
            ns, kind, item, cache, cache_age)
        if kind == "pr":
            base_sha = fairy.get_pr_base_sha(item)
            head_sha = fairy.get_pr_head_sha(item)
            _fetch_pr_head(ns, item["number"], head_sha)
            live = [s for s in
                    fairy.effective_review_states(reviews).values()
                    if not s.stale]
            status = {
                "state": ("merged" if forge_gcli.pr_merged(item)
                          else str(item.get("state") or "")),
                "auto_merge": fairy.get_auto_merge_info(ns, item,
                                                        timeline=timeline),
                "approvals": sum(s.state == "APPROVED" for s in live),
                "change_requests": sum(s.state == "CHANGES_REQUESTED"
                                       for s in live),
                "base_sha": base_sha,
                "head_sha": head_sha,
                "patch_ids": _patch_ids(db, ns, kind, token, base_sha, head_sha),
            }
        else:
            status = {
                "state": str(item.get("state") or ""),
                "labels": labels(item),
            }
        discussion = fairy.build_llm_discussion(
            reviews, comments, review_comments, timeline)
        last_change = next((d["state"] for d in reversed(discussion)
                            if d["kind"] == "state"), None)
        if (status["state"] in ("closed", "merged")) \
                != (last_change in ("closed", "merged")):
            logger.warning("%s #%s: the forge says %s but the timeline's "
                           "last state change is %s", kind, token,
                           status["state"], last_change)
        snapshot = {
            "title": str(item.get("title") or ""),
            "author": fairy.get_pr_author(item),
            "body": str(item.get("body") or ""),
            "html_url": str(item.get("html_url") or ""),
            "updated_at": item.get("updated_at"),
            **status,
            "discussion": discussion,
        }
        db.push(filedb.ITEM_STATE, kind, token, snapshot)
    except Exception as exc:
        logger.warning("%s #%s: item snapshot not refreshed: %s",
                       kind, token, exc)
        return None
    if ns.halt_keyword:
        for post in (item, *comments, *reviews, *review_comments):
            if ns.halt_keyword in str(post.get("body") or ""):
                reason = (f"{fairy.get_pr_author(post)} posted "
                          f"{ns.halt_keyword!r} in "
                          f"{post.get('html_url') or snapshot['html_url']}")
                halt_marker.halt(halt_marker.path(db.root, ns.halt_file), reason)
                raise Halted(reason)
    return snapshot


def _refresh_activity(db: filedb.Db, state: str, kind: str,
                      token: filedb.TicketId,
                      prepared: fairy.Decision | fairy.PreparedPR
                      | issue_fairy.PreparedIssue) -> None:
    """Bring a standing ticket's activity stamp -- the age the TUI
    shows -- up to date with the item, leaving everything else the
    ticket says as it is."""
    if prepared.last_activity is None:
        return
    stamp = prepared.last_activity.isoformat()
    db.try_move(state, state, kind, token,
                mutate=lambda d: d.update(last_activity_iso=stamp))


def _route(db: filedb.Db, kind: str, number: filedb.TicketId, state: str,
           data: dict, prior: str | None) -> None:
    """Scan-time routing: dst-first, and refused when the item moved at
    all (worker claim, operator y/s/x) during the seconds the prepare
    took -- the scan's decision was made against ``prior`` and is stale
    for anything else."""
    if not db.replace(state, kind, number, data, expect=prior):
        logger.info("%s #%s moved while preparing; not rerouted",
                    kind, number)


def scan_side(db: filedb.Db, ns: argparse.Namespace, kind: str,
              items: list[dict], *, now: datetime, cache,
              self_login: Callable[[], str | None],
              forced: set[filedb.TicketId], refetch: set[int] = frozenset(),
              serve_operator: Callable[[], None] = lambda: None,
              ) -> set[tuple[str, int]]:
    """One gate pass over ``items``, the side's open listing, plus the
    forced numbers and the ``refetch`` numbers not among them, fetched
    one by one; returns the open set. ``self_login`` is asked for the
    agent's login only when there is something to gate.
    ``serve_operator`` runs before every item, so a y or r pressed
    during the pass is answered without waiting for its end."""
    if kind == "pr":
        fetch_one, forced_ns = fairy.get_pr, ns.force_review_prs
        wip_re = fairy.compile_wip_regex(
            fairy.DEFAULT_WIP_PREFIXES + (ns.wip_prefixes or []))
    else:
        fetch_one, forced_ns = issue_fairy.get_issue, ns.force_review_issues
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
    listed = {int(i["number"]) for i in items if str(i.get("number")).isdigit()}
    # forced/requested numbers may be closed or merged: absent from
    # the open listing but explicitly asked for
    missing = sorted((forced_ns | forced_base | refetch) - listed)
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
                    self_login=self_login() if items else None,
                    forced_ns=forced_ns,
                    wip_re=wip_re if kind == "pr" else None,
                    cache_age=cache_age, limit=limit, evals=evals,
                    template_only=template_only,
                    serve_operator=serve_operator)
    finally:
        forced_ns -= added_forced
    return open_set


def snapshot_closed(db: filedb.Db, ns: argparse.Namespace, kind: str,
                    closed_items: list[dict], cache,
                    serve_operator: Callable[[], None]) -> str | None:
    """Refresh the listed closed items' snapshots, skipping those whose
    stored snapshot already carries the listed updated_at. Returns the
    oldest updated_at among the items whose snapshot failed, so that
    the next listing reaches them again; None when none failed."""
    # --scan-closed-days: snapshot-only visibility. Deliberately
    # NOT fed to the gates and NOT part of the open set: a closed
    # ticket must never become a review candidate by mere listing,
    # and closure cancels/pruning must proceed as if the option were
    # off. The infinite cache age skips the edit-catching discussion
    # TTL for these items only.
    failed: list[str] = []
    for item in closed_items:
        serve_operator()
        token = str(item.get("number"))
        if not token.isdigit() or not item.get("updated_at"):
            continue
        stored = db.get(filedb.ITEM_STATE, kind, token)
        if not (stored and stored.get("updated_at") == item.get("updated_at")) \
                and not _put_snapshot(db, ns, kind, token, item, cache,
                                      timedelta.max):
            failed.append(str(item["updated_at"]))
    return min(failed, key=iso_to_dt, default=None)


def _unacknowledged_agent_post_at(prior: str | None,
                                 prior_data: dict | None) -> str | None:
    """When the agent posted on the item on its own and no y or s has
    acknowledged that yet: the posted_at, carried by every ticket that
    replaced the posted/ one since. None otherwise."""
    if not prior_data:
        return None
    if prior == "posted" and prior_data.get("agent_promoted") \
            and not prior_data.get("acknowledged_at"):
        return prior_data.get("posted_at") or prior_data.get("state_changed_at")
    return prior_data.get("unacknowledged_post_at")


def _scan_items(db, ns, kind, items, *, now, cache, self_login, forced_ns,
                wip_re, cache_age, limit, evals={},
                template_only=frozenset(),
                serve_operator: Callable[[], None] = lambda: None) -> None:
    queued = 0
    for item in sorted(items, key=lambda i: (int(i["number"]) not in forced_ns,
                                             int(i["number"]))):
        serve_operator()
        number = int(item["number"])
        token = str(number)
        snapshot = _put_snapshot(db, ns, kind, token, item, cache, cache_age)
        prior = db.find(kind, token)
        if prior in IN_FLIGHT:
            continue
        prior_data = db.get(prior, kind, token) if prior else None
        agent_post_at = _unacknowledged_agent_post_at(prior, prior_data)
        carried = {"unacknowledged_post_at": agent_post_at} if agent_post_at else {}
        if prior == "reviewed" and number not in forced_ns and prior_data:
            # Any guard-matching verdict stands -- including skips that
            # carry label changes: they sit in reviewed/ awaiting the
            # operator, and a re-queue would burn an LLM run and yank
            # the row out from under the cursor every scan.
            if kind == "pr" and prior_data.get("expected_head_sha") \
                    != fairy.get_pr_head_sha(item):
                stands = fairy.only_rebased_since(
                    snapshot, prior_data.get("expected_updated_at"),
                    prior_data.get("expected_patch_ids"))
            else:
                stands = prior_data.get("expected_updated_at") == item.get("updated_at")
            if stands:
                continue  # standing verdict; reuse
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
            # wait is served; a moved updated_at is judged after
            # prepare (below), because only the discussion tells label
            # edits apart from real activity.
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
        error_backoff_h = 0.0
        if prior == "error" and prior_data and number not in forced_ns:
            wait = backoff_wait_h(prior_data.get("error_backoff_h", 0))
            # served waits double like the skip backoff: a 100%-failing
            # item costs log2, not linear, retries until someone looks
            if _age_h(prior_data, now) < wait:
                continue  # a persistently failing item must not burn spend every cycle
            error_backoff_h = wait
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
                # line and doubling backoff pacing; the archive and a
                # standing verdict outrank it
                if prior not in ("posted", "cancelled", "reviewed"):
                    _route(db, kind, token, "error", {
                        "title": str(item.get("title") or ""),
                        "html_url": str(item.get("html_url") or ""),
                        "error": f"prepare failed: {exc}",
                        "error_backoff_h": error_backoff_h,
                        "expected_updated_at": item.get("updated_at"),
                    }, prior)
                continue
        if prior == "reviewed" and not ns.auto_mode and number not in forced_ns:
            # A reviewed/ verdict in manual mode is the operator's
            # case, changed item or not: requeueing here replaced
            # their verdict with whatever the fresh round produced
            # (production: #23863 after a blocked y, #21117 before
            # any y -- genuine data loss). A stale y is caught by
            # the send guard; only auto mode re-reviews on change.
            _refresh_activity(db, prior, kind, token, prepared)
            continue
        if isinstance(prepared, fairy.Decision):
            state = gate_state(prepared)
            # A plain gate skip must not clobber the archive: posted/
            # cancelled/error records outrank "nothing to do today".
            if state == "skipped" and prior in ("posted", "cancelled", "error",
                                                "skipped"):
                if prior != "skipped" or (prior_data or {}).get("llm_at"):
                    _refresh_activity(db, prior, kind, token, prepared)
                    continue
            # A verdict the agent posted on its own outranks the
            # attention rows until the operator's y or s: rerouted to
            # merge-ready/ it would vanish among them before anyone
            # read it.
            if prior == "posted" and agent_post_at:
                _refresh_activity(db, prior, kind, token, prepared)
                continue
            ticket = {**gate_ticket(prepared, item), **carried}
            if state == "error":
                if prior in ("posted", "cancelled", "reviewed"):
                    # never clobber archive or a standing verdict
                    _refresh_activity(db, prior, kind, token, prepared)
                    continue
                ticket["error"] = prepared.reason
                ticket["error_backoff_h"] = error_backoff_h
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
            if last_iso == prior_data.get("reviewed_activity_iso") \
                    and (kind != "pr" or prior_data.get("expected_head_sha")
                         == fairy.get_pr_head_sha(item)):
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
            "body": str(item.get("body") or ""),
            "head_branch": fairy.get_pr_head_branch(item),
            # queued/error tickets need the guard too, or an operator x
            # on them cannot stick until new activity
            "expected_updated_at": item.get("updated_at"),
            "html_url": str(getattr(prepared, "pr", getattr(prepared, "issue", {})).get("html_url") or ""),
            "skip_backoff_h": backoff_h,
            "error_backoff_h": error_backoff_h,
            "forced": number in forced_ns,
            "prepared": fairy.prepared_to_dict(prepared),
            **carried,
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


REASON_MERGED = "merged"
REASON_CLOSED_UNMERGED = "closed without merge"
REASON_CLOSED = "closed"


def ingest_item(db: filedb.Db, ns: argparse.Namespace, kind: str,
                number: filedb.TicketId, cache) -> tuple[dict, dict | None]:
    """Fetch the item behind ``number`` and refresh its filedb snapshot
    with it: every forge read of an item lands in items/. Returns the
    item and the snapshot, None when that could not be refreshed."""
    forge_number = filedb.forge_number(number)
    item = fairy.get_pr(ns, forge_number) if kind == "pr" \
        else issue_fairy.get_issue(ns, forge_number)
    snapshot = _put_snapshot(db, ns, kind, str(forge_number), item, cache,
                             timedelta(hours=ns.discussion_cache_max_age_hours))
    return item, snapshot


def closure_reason_of(kind: str, item: dict) -> str | None:
    """Why the item is not open: "merged" is the success story and must
    not read as a failure in the UI (production: #23913 showed plain
    cancelled after the operator merged it). None while it is open."""
    if kind == "pr" and forge_gcli.pr_merged(item):
        return REASON_MERGED
    if item.get("state") == "closed":
        return REASON_CLOSED_UNMERGED if kind == "pr" else REASON_CLOSED
    return None


def closure_reason(db: filedb.Db, ns: argparse.Namespace, kind: str,
                   number: filedb.TicketId, cache) -> str | None:
    """One fetch to name why an item left the open listing. None means
    the item is in fact still open -- the listing was transiently
    short -- and must not be cancelled at all. A failed fetch keeps the
    old revivable "not open"."""
    try:
        item, _ = ingest_item(db, ns, kind, number, cache)
    except Exception as exc:
        logger.warning("%s #%s left the listing but the fate fetch "
                       "failed: %s", kind, number, exc)
        return "not open"
    return closure_reason_of(kind, item)


def cancel_tickets(db: filedb.Db, kind: str,
                   closure: Callable[[int], str | None]) -> None:
    """Cancel the live tickets of ``kind`` whose forge number
    ``closure`` gives a reason for; None keeps the ticket."""
    # attention tickets too: a merged PR's merge-ready/ci-blocked row
    # would otherwise sit there forever (prune skips non-settled states)
    for state in ("queued", "reviewed", "ci-blocked", "merge-ready",
                  "awaiting-approver"):
        for k, number in db.list_state(state):
            if k != kind:
                continue
            reason = closure(filedb.forge_number(number))
            if reason is None:
                continue
            # try_move: a claimed item's lock is held for the whole
            # review and must not stall the pass; retried next scan
            if db.try_move(state, "cancelled", kind, number,
                           mutate=lambda d, r=reason: d.update(reason=r)):
                logger.info("%s #%s cancelled: %s", kind, number, reason)


def ci_pending_numbers(db: filedb.Db, kind: str) -> set[int]:
    """Forge numbers whose gate ticket says the head's CI was still
    running when the item was last looked at."""
    return {filedb.forge_number(number)
            for state in ("skipped", "ci-blocked", "merge-ready", "awaiting-approver")
            for k, number in db.list_state(state)
            if k == kind and (db.get(state, k, number) or {}).get("ci_pending")}


class SideCaches:
    """The sides' gcli caches, each loaded on first use and saved when
    the holder closes, so a pass touches only the files it needs."""

    def __init__(self, sides: dict[str, argparse.Namespace]) -> None:
        self.sides = sides
        self.loaded: dict[str, gcli_cache.Cache] = {}

    def __getitem__(self, kind: str) -> gcli_cache.Cache:
        if kind not in self.loaded:
            self.loaded[kind] = gcli_cache.load_cache(self.sides[kind].cache)
        return self.loaded[kind]

    def __enter__(self) -> SideCaches:
        return self

    def __exit__(self, *exc) -> None:
        for kind, cache in self.loaded.items():
            gcli_cache.save_cache(self.sides[kind].cache, cache)


def sides_of(pr_ns: argparse.Namespace | None,
             issue_ns: argparse.Namespace | None
             ) -> dict[str, argparse.Namespace]:
    return {kind: ns for kind, ns in (("pr", pr_ns), ("issue", issue_ns))
            if ns is not None}


SCAN_STAMPS_DOC = "scan-stamps.json"
SNAPSHOT_RETRY_DOC = "snapshot-retry.json"
LISTING_OVERLAP = timedelta(minutes=5)


def listing_cutoff(ns: argparse.Namespace, stamp: str | None,
                   now: datetime) -> datetime:
    """Where the side's listing stops: LISTING_OVERLAP before ``stamp``,
    the newest updated_at the last successful listing saw, on the
    assumption that an update visible after a listing carries an
    updated_at no older than that minus the overlap. Never before the
    --scan-closed-days window; ``now`` (the simulated clock under
    --simulate-past) when neither bounds it."""
    now = getattr(ns, "simulate_past", None) or now
    bounds = [now - timedelta(days=ns.scan_closed_days)] \
        if ns.scan_closed_days > 0 else []
    if (since := iso_to_dt(stamp)) is not None:
        bounds.append(since - LISTING_OVERLAP)
    return max(bounds, default=now)


def scan_pass(db: filedb.Db, pr_ns: argparse.Namespace | None,
              issue_ns: argparse.Namespace | None,
              now: datetime | None = None,
              dry_run: bool = False, *, full: bool = True) -> None:
    """One scan of every side. A full pass lists all open items and
    gates each, cancels the tickets of items no longer listed and
    prunes; an incremental pass gates only the listed open items whose
    snapshot is behind the listing or which have no ticket (a review
    --limit deferred), plus the tickets whose CI was pending, and
    cancels from the listing. Every pass lists what the
    forge updated since the side's stamp, the newest updated_at its
    last successful listing saw; the stamp is recorded once the
    listing succeeded, so a failed listing is redone from the same
    cutoff. A closed snapshot that failed is remembered by its
    updated_at, and the full pass lists from there so that it is
    tried again. A side without a stamp yet is scanned like a full
    pass."""
    now = now or datetime.now(timezone.utc)
    forced = consume_requests(db)
    stamps_doc = db.root / SCAN_STAMPS_DOC
    stamps: dict[str, str] = db.read(stamps_doc) or {}
    retry_doc = db.root / SNAPSHOT_RETRY_DOC
    retry: dict[str, str] = db.read(retry_doc) or {}
    open_set: set[tuple[str, int]] = set()
    kinds: set[str] = set()
    # A --forced-only side's open_set is just the named items, not the
    # open listing: everything else would look closed, so closing and
    # pruning are skipped for it.
    full_kinds: set[str] = set()
    sides = sides_of(pr_ns, issue_ns)
    served: dict[str, set[filedb.TicketId]] = {kind: set() for kind in sides}
    logins: dict[str, str | None] = {}

    def login(kind: str) -> str | None:
        if kind not in logins:
            logins[kind] = forge_gcli.self_login(sides[kind])
        return logins[kind]

    with SideCaches(sides) as caches:
        def serve_outgoing() -> None:
            send_outgoing(db, sides, caches, dry_run=dry_run)

        def serve_operator() -> None:
            """Between two scanned items: post the y presses and answer
            the requests that arrived since the pass consumed its set,
            each by fetching just the named item."""
            serve_outgoing()
            for kind, ns in sides.items():
                late = {n for k, n in db.list_state("requests")
                        if k == kind} - served[kind]
                if not late:
                    continue
                served[kind] |= late
                open_set.update(scan_side(
                    db, ns, kind, [], now=now, cache=caches[kind],
                    self_login=partial(login, kind), forced=late,
                    serve_operator=serve_outgoing))
                finish_requests(db, {kind: late}, {kind})
        for kind, ns in sides.items():
            kinds.add(kind)
            if not ns.forced_only:
                full_kinds.add(kind)
            pending = forced[kind] - served[kind]
            served[kind] |= forced[kind]
            if ns.forced_only:
                open_set |= scan_side(
                    db, ns, kind, [], now=now, cache=caches[kind],
                    self_login=partial(login, kind), forced=pending,
                    serve_operator=serve_operator)
                continue
            full_side = full or not stamps.get(kind)
            cutoff = listing_cutoff(
                ns, retry.get(kind) if full_side and retry.get(kind) else stamps.get(kind),
                now)
            listed = True
            try:
                changed = [
                    i for i in (forge_gcli.list_since(ns, "pulls", cutoff, state="all")
                                if kind == "pr"
                                else issue_fairy.list_issues_since(ns, cutoff))
                    if str(i.get("number")).isdigit()]
            except Exception as exc:
                logger.warning("%s: listing the items updated since %s failed; "
                               "what they need waits for the next pass: %s",
                               kind, cutoff, exc)
                changed, listed = [], False
            if changed:
                logger.info("%s: %d item(s) updated since %s", kind, len(changed), cutoff)
            reasons = {int(i["number"]): closure_reason_of(kind, i) for i in changed}
            closed = [i for i in changed if reasons[int(i["number"])]]
            if full_side:
                items = fairy.list_open_prs(ns) if kind == "pr" \
                    else issue_fairy.list_open_issues(ns)
            else:
                items = [i for i in changed if not reasons[int(i["number"])]
                         and ((db.get(filedb.ITEM_STATE, kind, str(i["number"])) or {})
                              .get("updated_at") != i.get("updated_at")
                              or db.find(kind, str(i["number"])) is None)]
            refetch = set() if full_side else ci_pending_numbers(db, kind) - {
                int(i["number"]) for i in items + closed}
            open_set |= scan_side(
                db, ns, kind, items, now=now, cache=caches[kind],
                self_login=partial(login, kind), forced=pending, refetch=refetch,
                serve_operator=serve_operator)
            failed = snapshot_closed(db, ns, kind, closed, caches[kind],
                                     serve_operator) \
                if ns.scan_closed_days > 0 else None
            cancel_tickets(db, kind, lambda n: None if (kind, n) in open_set
                           else reasons.get(n) or (closure_reason(
                               db, ns, kind, str(n), caches[kind]) if full_side else None))
            if listed:
                newest = str(changed[0]["updated_at"]) if changed \
                    else forge_gcli.newest_updated_at(ns, kind)
                if newest:
                    stamps[kind] = newest
                if failed:
                    retry[kind] = min((s for s in (failed, retry.get(kind)) if s),
                                      key=iso_to_dt)
                elif full_side:
                    retry.pop(kind, None)
        finish_requests(db, forced, kinds)
    db.write(stamps_doc, stamps)
    db.write(retry_doc, retry)
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
    if full and full_kinds == kinds:  # prune's keep-set is kind-blind
        for ns, kind in ((pr_ns, "pr"), (issue_ns, "issue")):
            if ns is None:
                continue
            before = now - timedelta(days=ns.workset_retention_days)
            for state in ("posted", "skipped", "cancelled", "error",
                          filedb.SUPERSEDED_STATE):
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
            for c in review.get("label_changes") or () if isinstance(c, dict)),
        branches=fairy.parse_branch_records(review.get("branches")))
    if kind == "pr":
        decision = fairy.decision_from_review(
            llm, number=filedb.forge_number(number),
            title=ticket.get("title", ""),
            author=ticket.get("author", ""),
            auto_merge=ticket.get("auto_merge", "-"),
            last_activity=iso_to_dt(ticket.get("last_activity_iso")),
            base_reason="persisted review",
            reviewer_username=ticket.get("reviewer"))
    else:
        decision = issue_fairy.issue_review_decision(
            llm, number=filedb.forge_number(number),
            title=ticket.get("title", ""),
            author=ticket.get("author", ""), reason="persisted review",
            last_activity=iso_to_dt(ticket.get("last_activity_iso")))
    return dataclasses_replace(
        decision,
        expected_pr_updated_at=ticket.get("expected_updated_at"),
        expected_head_sha=ticket.get("expected_head_sha"),
        expected_patch_ids=(None if ticket.get("expected_patch_ids") is None
                            else tuple(ticket["expected_patch_ids"])))


def postable(decision: fairy.Decision | None) -> bool:
    return decision is not None and (
        decision.action in fairy.ACTIONABLE_DECISIONS
        or fairy.decision_has_label_changes(decision))


def staleness_reason(ns: argparse.Namespace, kind: str, item: dict,
                     decision: fairy.Decision,
                     snapshot: dict | None) -> str | None:
    """Why ``decision`` must not be posted onto ``item`` as it is now:
    the item moved since the review, or is no longer open. A PR's
    ``snapshot`` (its refreshed items/ record) tells a push that only
    rebased the series, which does not count as moving."""
    if kind == "pr":
        return fairy.check_pr_still_unchanged(ns, item, decision, snapshot)
    return issue_fairy.check_issue_still_unchanged(ns, item, decision)


def post_decision(ns: argparse.Namespace, kind: str, item: dict,
                  decision: fairy.Decision, *, cache,
                  counts: dict[str, int]) -> str | None:
    """The forge side effects, through fairy/issue_fairy's submit
    seams; the block reason when one refused, None when everything
    went out."""
    if kind == "issue":
        return issue_fairy.submit_issue_decision(
            ns, item, decision, cache=cache, submitted_counts=counts)
    if decision.action in fairy.ACTIONABLE_DECISIONS:
        reason = fairy.submit_decision_action(ns, decision, cache=cache)
        if reason is not None:
            return reason
        counts[decision.action] += 1
    if fairy.decision_has_label_changes(decision):
        fairy.apply_triage_labels(ns, item, decision)
    return None


def link_ticket_branches(ns: argparse.Namespace, ticket: dict) -> None:
    """Link the pushed branches in the ticket's review message to their
    forge pages; a message edited by hand or reviewed before the
    linking pass gets them at send time."""
    review = ticket.get("review")
    if not review:
        return
    review["message"] = fairy.link_published_branches(
        ns, review.get("message", ""),
        fairy.parse_branch_records(review.get("branches")),
        str(ticket.get("html_url") or ""))


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
        link_ticket_branches(ns, ticket)
        decision = ticket_decision(kind, number, ticket)
        if not postable(decision):
            claim.finish("reviewed", dict(ticket, send_blocked="nothing to post"))
            return "reviewed"
        if dry_run:
            logger.info("%s #%s: DRY RUN, would post: %s", kind, number,
                        fairy.manual_action_description(decision))
            claim.abort()
            return None
        item, snapshot = ingest_item(db, ns, kind, number, cache)
        # the operator's Y: post as-is although the item may have
        # moved since the review; popped so the archive stays clean
        reason = None if ticket.pop("force_post", None) \
            else staleness_reason(ns, kind, item, decision, snapshot)
        if reason is None:
            reason = post_decision(ns, kind, item, decision, cache=cache,
                                   counts=counts)
        if reason is None:
            claim.finish("posted", dict(
                ticket, posted_at=datetime.now(timezone.utc).isoformat()))
            logger.info("%s #%s posted: %s", kind, number,
                        fairy.manual_action_description(decision))
            return "posted"
        if ticket.pop("agent_promoted", None) and ns.auto_mode:
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
    except BaseException:
        claim.abort()
        raise


def agent_promotes(db: filedb.Db, ns: argparse.Namespace, kind: str,
                   number: filedb.TicketId, ticket: dict) -> bool:
    """Whether the send pass, not an operator, sends this reviewed/
    verdict: every base verdict under --auto-mode, under
    --vetted-auto-mode those the wrapper's vetter passed. Never a
    sample evaluation, never an item that has any (the operator who
    asked for several reviews picks the one to post) and never a
    verdict a blocked send parked: that waits for Y, r or s."""
    if not filedb.is_base(number) or db.evaluations(kind, number) \
            or ticket.get("send_blocked"):
        return False
    vetting = ticket.get("vetting") or {}
    return ns.auto_mode or (
        ns.vetted_auto_mode and vetting.get("hold_for_human_inspection") is False)


def promote_reviewed(db: filedb.Db, ns: argparse.Namespace, kind: str, *,
                     dry_run: bool) -> None:
    """Move the standing verdicts the side's mode lets the agent post
    to outgoing/, stamped agent_promoted; a dry run only logs them,
    since a promotion is persistent staging a later normal run would
    post."""
    for k, number in db.list_state("reviewed"):
        # find() precedence: a reviewed/ crash remnant behind a later
        # state must not be promoted (and posted) a second time
        if k != kind or db.find(kind, number) != "reviewed":
            continue
        ticket = db.get("reviewed", kind, number) or {}
        if not (postable(ticket_decision(kind, number, ticket))
                and agent_promotes(db, ns, kind, number, ticket)):
            continue
        if dry_run:
            logger.info("%s #%s: DRY RUN, would promote to outgoing/", kind, number)
        else:
            db.try_move("reviewed", "outgoing", kind, number, mutate=promote)


def promote(ticket: dict) -> None:
    """The agent's own post: flagged for the operator's y or s, and it
    supersedes the earlier unacknowledged post the ticket carried."""
    ticket.pop("unacknowledged_post_at", None)
    ticket["agent_promoted"] = True


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


def send_outgoing(db: filedb.Db, sides: dict[str, argparse.Namespace],
                  caches: SideCaches, *, dry_run: bool = False) -> None:
    """Post every outgoing/ ticket through the sides' open caches."""
    for kind, ns in sides.items():
        outgoing = [n for k, n in db.list_state("outgoing") if k == kind]
        if not outgoing:
            continue
        counts = {action: 0 for action in fairy.ACTIONABLE_DECISIONS}
        for number in outgoing:
            try:
                send_one(db, ns, kind, number, cache=caches[kind],
                         counts=counts, dry_run=dry_run)
            except Exception:
                logger.exception("%s #%s: send failed; stays in outgoing/",
                                 kind, number)


def send_pass(db: filedb.Db, sides: dict[str, argparse.Namespace],
              *, dry_run: bool = False) -> None:
    for kind, ns in sides.items():
        if ns.auto_mode or ns.vetted_auto_mode:
            promote_reviewed(db, ns, kind, dry_run=dry_run)
    with SideCaches(sides) as caches:
        send_outgoing(db, sides, caches, dry_run=dry_run)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        add_help=False,
        description="Repo agent: scan the forge and maintain the filedb tickets "
                    "for one repository's PRs and issues.",
        epilog=OVERRIDE_EPILOG,
    )
    add_grouped_help(p, _help)
    p.add_argument("--db-root", type=Path, required=True,
                   help="filedb root; its config.toml, written by "
                        "configurator.py, carries the side options")
    p.add_argument("--loop", type=float, default=0, metavar="SECONDS",
                   help="tick every N seconds: one listing per side tells "
                        "what moved on the forge, a rescan of that follows, "
                        "of everything hourly; operator files "
                        "(requests/, outgoing/) and a worker's verdict "
                        "(reviewed/) wake the loop instantly via watchdog "
                        "(default: one full pass, cron style)")
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
    return p


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
             args: argparse.Namespace, *, full: bool = True) -> None:
    sides = sides_of(pr_ns, issue_ns)

    def review_cycle() -> None:
        scan_pass(db, pr_ns, issue_ns, dry_run=args.dry_run, full=full)
        if args.drain:
            worker.drain(db, sides, parallel=args.drain)

    review_cycle()
    if args.ask:
        ask_pass(db, set(sides),
                 retry=review_cycle if args.drain else None)
    send_pass(db, sides, dry_run=args.dry_run)
    log_summary(db)


PASS_RETRY_S = 60.0
FULL_PASS_S = 3600.0


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
    one_pass(db, _forced_only(pr_ns), _forced_only(issue_ns), args, full=False)


def _help() -> str:
    pr_agent = fairy.make_parser(worker=False)
    issue_agent = issue_fairy.make_parser(worker=False)
    exec_common, exec_pr, exec_issue = split_side_actions(
        side_actions(fairy.make_parser(), minus=pr_agent),
        side_actions(issue_fairy.make_parser(), minus=issue_agent))
    return grouped_help(
        make_parser(),
        config_option_groups(pr_agent, issue_agent)
        + [("review execution options, both sides (the worker's; the "
            "agent runs them under --drain)", exec_common),
           ("review execution options, PR side", exec_pr),
           ("review execution options, issue side", exec_issue)])


def main() -> int:
    args, pr_over, issue_over, sections = parse_scoped_overrides(
        sys.argv[1:], make_parser(),
        fairy.make_parser(), fairy.make_parser(),
        issue_fairy.make_parser(), issue_fairy.make_parser())
    pr_opts, issue_opts = db_config.read_side_options(args.db_root)
    pr_ns = fairy.parse_args(options_argv({**pr_opts, **pr_over})) \
        if pr_opts is not None else None
    issue_ns = issue_fairy.parse_args(
        options_argv({**issue_opts, **issue_over})) \
        if issue_opts is not None else None
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
    fairy.validate_sides(pr_ns, issue_ns)
    db = filedb.Db(args.db_root)
    logger.info("agent for %s/%s, db %s", lead.owner, lead.repo, db.root)
    db_config.log_side_argv(
        options_argv(pr_opts) if pr_opts is not None else None,
        options_argv(issue_opts) if issue_opts is not None else None,
        options_argv(pr_over)
        if pr_opts is not None or "--prs" in sections else [],
        options_argv(issue_over)
        if issue_opts is not None or "--issues" in sections else [])
    for ns in (pr_ns, issue_ns):
        if ns is not None and getattr(ns, "simulate_past", None):
            warn_simulate_past_limitations(ns.simulate_past)
    # The forge rescan stays on the --loop interval, but operator files
    # must not wait for it: a request or a y-press (outgoing/) wakes the
    # loop within milliseconds; a wake without a request only needs the
    # send pass, not a full forge scan.
    wake = Event()
    watched = watch_paths([db.root / "requests", db.root / "outgoing",
                           db.root / "reviewed"],
                          wake.set) is not None
    sides = sides_of(pr_ns, issue_ns)
    next_scan = next_full = 0.0

    def scan_due() -> bool | None:
        """None until the next tick; then whether the due pass is a
        full one (always without --loop, else hourly)."""
        nonlocal next_scan
        if time.monotonic() < next_scan:
            return None
        next_scan = time.monotonic() + args.loop
        return not args.loop or time.monotonic() >= next_full

    marker = halt_marker.path(db.root, lead.halt_file)
    while True:
        try:
            if (halt := halt_marker.reason(marker)) is not None:
                logger.warning("halted: %s -- remove %s to resume",
                               halt, marker)
                next_scan = time.monotonic() + args.loop
            elif (full := scan_due()) is not None:
                one_pass(db, pr_ns, issue_ns, args, full=full)
                if full:
                    next_full = time.monotonic() + FULL_PASS_S
            elif db.list_state("requests"):
                requests_pass(db, pr_ns, issue_ns, args)
            else:
                send_pass(db, sides, dry_run=args.dry_run)
        except Halted as halt:
            if not args.loop:
                raise
            logger.error("%s", halt)
        except Exception:
            # A transient forge/gcli error must not kill the daemon;
            # one-shot (cron) mode still fails loudly via its exit code.
            if not args.loop:
                raise
            retry_s = min(PASS_RETRY_S, args.loop)
            logger.exception("pass failed; retrying in %gs", retry_s)
            wake.wait(retry_s)
            wake.clear()
            continue
        if not args.loop:
            return 0
        wait_s = next_scan - time.monotonic()
        wake.wait(max(0.0, wait_s if watched
                      else min(wait_s, WATCH_FALLBACK_POLL_S)))
        wake.clear()


if __name__ == "__main__":
    raise SystemExit(main())
