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

The LLM worker: claims queued filedb tickets and turns them into
verdicts. The flock held on each claim is its liveness signal (a
dead worker's claims are reaped by the agent). Parallelism composes
both ways: --parallel N
reviews N tickets concurrently in one process, and several worker
processes arbitrate through the same claim protocol. The ticket
carries the full prepared payload, so the worker never talks to the
forge; the wrapper's stage notes land in the claimed ticket via
--workset-file.

Verdict routing: actionable reviews (and skips that still carry label
changes) go to reviewed/ for the operator; plain LLM skips go to
skipped/ carrying the ticket's doubled backoff; errors go to error/.

What belongs here: the claim loop, wrapper invocation glue and
verdict routing. What does NOT belong: gates and ticket creation
(agent), file atomicity (filedb), posting (the agent's send pass).
"""

from __future__ import annotations

import argparse
import logging
import shlex
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from datetime import datetime, timezone
from pathlib import Path

import agent
import fairy
import filedb
import issue_fairy
import workset
from common import add_file_log, setup_logging, watch_paths

__all__ = ["main", "review_claim", "drain"]

logger = logging.getLogger(__name__)


def verdict_state(decision: fairy.Decision) -> str:
    if decision.llm_classification == "error":
        return "error"
    if decision.llm_classification == "skip" and not decision.label_changes:
        return "skipped"
    return "reviewed"


def verdict_fields(decision: fairy.Decision, prepared) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    item = getattr(prepared, "pr", None) or getattr(prepared, "issue", {})
    if decision.llm_classification == "error":
        # the guard makes an operator x on the error row stick
        return {"error": decision.reason, "llm_at": now,
                "expected_updated_at": item.get("updated_at")}
    return {
        "error": None,
        "review": {
            "classification": decision.llm_classification,
            "message": decision.llm_message,
            "label_changes": [
                {"label": c.label, "op": c.op, "reason": c.reason, "post": c.post}
                for c in decision.label_changes
            ],
        },
        "action": decision.action,
        "reason": decision.reason,
        "expected_updated_at": item.get("updated_at"),
        "expected_head_ref": (fairy.get_pr_head_ref(item)
                              if getattr(prepared, "pr", None) is not None else None),
        "last_activity_iso": (decision.last_activity.isoformat()
                              if decision.last_activity else None),
        "llm_at": now,
    }


def review_claim(claim: filedb.Claim, ns: argparse.Namespace) -> str:
    ticket = claim.read()
    if claim.kind == "pr":
        prepared = fairy.prepared_pr_from_dict(ticket["prepared"])
    else:
        prepared = issue_fairy.prepared_issue_from_dict(ticket["prepared"])
    ns.workset_file_override = str(claim.path)
    try:
        if claim.kind == "pr":
            decision = fairy.safe_apply_llm_review_to_prepared(ns, prepared)
        else:
            try:
                decision = issue_fairy.evaluate_issue(ns, prepared)
            except Exception as exc:
                decision = fairy.Decision(
                    claim.number, ticket.get("title", ""), ticket.get("author", ""),
                    "-", "error", str(exc), None, "error", "")
    finally:
        ns.workset_file_override = None
    ticket = claim.read()  # the wrapper noted stage/triage/drafts meanwhile
    ticket.pop("prepared", None)
    ticket.pop("stage", None)
    ticket.update(verdict_fields(decision, prepared))
    state = verdict_state(decision)
    claim.finish(state, ticket)
    # workset.update_json's sidecar lock next to the claimed file
    claim.path.with_suffix(".lock").unlink(missing_ok=True)
    logger.info("%s #%d -> %s (llm %s)", claim.kind, claim.number, state,
                decision.llm_classification)
    return state


def _drain_one(db: filedb.Db, sides: dict[str, argparse.Namespace],
               kind: str, number: int) -> bool | None:
    """Claim and review one ticket: True reviewed, False routed to
    error/, None when the claim was lost to another worker."""
    claim = db.claim("queued", "llm", kind, number)
    if claim is None:
        return None
    # a shallow copy per review: workset_file_override is per-ticket
    # state and the namespace is shared across drain threads
    ns = argparse.Namespace(**vars(sides[kind]))
    try:
        review_claim(claim, ns)
        return True
    except Exception as exc:
        # Aborting back to queued/ would re-claim the same
        # (sorted-first) ticket on every pass and starve the
        # worker; a ticket that cannot even be read belongs in
        # error/ where the agent's retry gate paces it.
        logger.exception("%s #%d: review failed; ticket -> error/",
                         kind, number)
        try:
            ticket = claim.read()
        except Exception:
            ticket = {}
        ticket.pop("prepared", None)
        ticket["error"] = f"worker: {exc}"
        ticket["llm_at"] = datetime.now(timezone.utc).isoformat()
        claim.finish("error", ticket)
        return False


def drain(db: filedb.Db, sides: dict[str, argparse.Namespace],
          parallel: int = 1) -> int:
    """Claim and review every queued ticket of the configured kinds, up
    to ``parallel`` at a time; returns the number reviewed. The claim
    protocol already arbitrates, so threads and other worker processes
    compose freely."""
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(parallel or 1))) as pool:
        while True:
            queued = [(k, n) for k, n in db.list_state("queued")
                      if k in sides]
            if not queued:
                return done
            outcomes = list(pool.map(
                lambda kn: _drain_one(db, sides, *kn), queued))
            done += outcomes.count(True)
            if all(o is None for o in outcomes):
                return done  # every claim lost: other workers own them


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM worker: claim queued filedb tickets and review them.",
    )
    p.add_argument("--pr-args", metavar="ARGS",
                   help="the PR side's argument string (models, wrapper, "
                        "hosts); ./fairy.py --help documents its contents")
    p.add_argument("--issue-args", metavar="ARGS",
                   help="the issue side's argument string; ./issue_fairy.py "
                        "--help documents its contents")
    p.add_argument("--db-root", type=Path,
                   help="filedb root (default: derived from the side's repo)")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="review up to N tickets concurrently (default: 1; "
                        "running several worker processes composes too)")
    p.add_argument("--loop", type=float, default=0, metavar="SECONDS",
                   help="keep waiting for tickets, rechecking every N seconds; "
                        "new queued/ files wake the worker instantly via "
                        "watchdog (default: drain and exit)")
    args = p.parse_args(argv)
    if not args.pr_args and not args.issue_args:
        p.error("at least one of --pr-args / --issue-args is required")
    return args


def main() -> int:
    args = parse_args()
    sides: dict[str, argparse.Namespace] = {}
    if args.pr_args:
        sides["pr"] = fairy.parse_args(shlex.split(args.pr_args))
    if args.issue_args:
        sides["issue"] = issue_fairy.parse_args(shlex.split(args.issue_args))
    lead = next(iter(sides.values()))
    setup_logging(fairy.logger, max(ns.verbose for ns in sides.values()),
                  logger, workset.logger, filedb.logger)
    for log_file in {ns.log_file for ns in sides.values() if ns.log_file}:
        add_file_log(log_file, fairy.logger, logger, workset.logger,
                     filedb.logger)
    db = filedb.Db(args.db_root or agent.db_root_for(lead))
    logger.info("worker for %s/%s, db %s", lead.owner, lead.repo, db.root)
    wake = Event()
    watch_paths([db.root / "queued"], wake.set)
    while True:
        try:
            drain(db, sides, parallel=args.parallel)
        except Exception:
            # same contract as the agent loop: a transient error must
            # not kill the daemon; one-shot mode fails loudly
            if not args.loop:
                raise
            logger.exception("drain failed; retrying in %gs", args.loop)
        if not args.loop:
            return 0
        wake.wait(args.loop)
        wake.clear()


if __name__ == "__main__":
    raise SystemExit(main())
