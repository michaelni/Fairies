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

Cost vs probability of losing a review run to a provider-ended turn.

The provider can end a reviewer's or the combiner's turn itself (e.g.
the "flagged for possible cybersecurity risk" content flag, see
``ProviderTurnFailed`` in llm_review_api.py). Retries happen on three
levels, all mirrored here:

  * ``run_parallel`` / ``review_pr`` retry the blocked stage; today's
    budget is one retry. This budget is the table's sweep variable.
  * ``call_llm_with_retries`` (fairy.py) re-runs the whole pipeline up
    to --llm-max-attempts times (default 3): one cycle.
  * When the whole cycle failed the item enters the ``error`` state and
    the agent re-queues it once the failure has aged past a doubling
    backoff: 24h, 48h, 96h, ... per served wait (agent.py
    backoff_wait_h), giving a permanently failing item about
    log2(--horizon-days) cycles within the horizon. --error-backoff
    fixed models a flat one-cycle-per-day pacing instead.

For each stage retry budget this prints:
  P(attempt)  one pipeline attempt fails on the block
  E[attempt]  expected cost of one pipeline attempt
  P(error)    the whole cycle fails: the item stalls in ``error`` a day
  E[cycle]    expected cost of one cycle -- attempts stop at the first
              success, a fully failed cycle burns all its attempts
  P(horizon)  every cycle of the horizon failed: the item is still
              unreviewed when the horizon ends
  E[total]    expected cost of all cycles up to the horizon
  dcost/dP    marginal E[total] per unit of P(error) removed by that
              budget's last retry (negative: the retry both saves money
              and probability, a stage retry being cheaper than the
              full-pipeline retries it avoids)

Model assumptions:
  * each reviewer has its own per-attempt block probability (a provider
    that never emits the flag, e.g. GLM so far, gets 0), the combiner
    its own; attempts block independently;
  * a reviewer that exhausts the budget is dropped; a pipeline attempt
    fails only when every reviewer is dropped or the combiner exhausts
    the budget; the combiner runs (and costs) only when a draft
    survives;
  * every attempt, blocked or not, is billed one full stage cost;
  * failure causes other than the block are out of scope, and only
    cost is modelled, not the wall-clock delay of retries.

Costs default to 1, so without :COST suffixes the cost columns are in
units of one stage attempt.

Usage:
  tools/turn_failed_retry_cost.py --reviewer 0.3 --combiner 0.3
  tools/turn_failed_retry_cost.py \\
      --reviewer code=0.10:0.9 --reviewer design=0.15:0.9 \\
      --reviewer glm=0 --combiner 0.10:1.4 --max-retries 6
"""

from __future__ import annotations

import argparse
import math


def attempt_outcome(reviewers: list[tuple[str, float, float]],
                    combiner: tuple[str, float, float] | None,
                    retries: int) -> tuple[float, float]:
    """One pipeline attempt at the given stage retry budget:
    (P(it fails on the block), its expected cost). ``reviewers`` and
    ``combiner`` are (label, block probability, attempt cost)."""
    dropped = [p ** (retries + 1) for _, p, _ in reviewers]
    all_dropped = math.prod(dropped)
    attempts = [sum(p ** i for i in range(retries + 1)) for _, p, _ in reviewers]
    fails = all_dropped
    cost = sum(c * a for (_, _, c), a in zip(reviewers, attempts))
    if combiner is not None:
        _, p, c = combiner
        fails += (1 - all_dropped) * p ** (retries + 1)
        cost += (1 - all_dropped) * c * sum(p ** i for i in range(retries + 1))
    return fails, cost


def retried_outcome(fails: float, cost: float,
                    attempts: int) -> tuple[float, float]:
    """Up to ``attempts`` pipeline attempts, stopping at the first
    success: (P(all fail), E[cost of the attempts made]).
    ``fails``/``cost`` are one attempt's, from attempt_outcome().
    One cycle is retried_outcome(fails, cost, --attempts); the whole
    horizon is retried_outcome(fails, cost, --attempts * cycles)."""
    if fails == 1:
        return 1.0, cost * attempts
    return fails ** attempts, cost * (1 - fails ** attempts) / (1 - fails)


def stage(text: str) -> tuple[str, float, float]:
    """[LABEL=]P[:COST] -> (label, block probability, attempt cost)"""
    label, _, spec = text.rpartition("=")
    prob, _, cost = spec.partition(":")
    try:
        value = float(prob)
        if not 0 <= value <= 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{prob!r}: not a probability in [0, 1]")
    return label, value, float(cost) if cost else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cost vs probability of losing a review run to a "
                    "provider-ended turn, per stage retry budget")
    parser.add_argument("--reviewer", type=stage, action="append",
                        required=True, metavar="[LABEL=]P[:COST]",
                        help="add a model reviewer with per-attempt block "
                             "probability P and attempt cost COST "
                             "(default 1); repeat per reviewer")
    parser.add_argument("--combiner", type=stage, metavar="[LABEL=]P[:COST]",
                        help="the combiner's block probability and attempt "
                             "cost (omit for a combiner-less single-reviewer "
                             "pipeline)")
    parser.add_argument("--attempts", type=int, default=3,
                        help="whole-pipeline attempts per cycle, "
                             "--llm-max-attempts (default: 3)")
    parser.add_argument("--horizon-days", type=int, default=365,
                        help="stop counting cycles past this age of the "
                             "item (default: 365)")
    parser.add_argument("--error-backoff", choices=("log2", "fixed"),
                        default="log2",
                        help="error re-queue cadence: waits doubling per "
                             "served wait as in agent.py, or a flat cycle "
                             "per day (default: log2)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="largest stage retry budget to tabulate "
                             "(default: 4)")
    args = parser.parse_args()

    cycles = (args.horizon_days if args.error_backoff == "fixed"
              else int(math.log2(args.horizon_days + 1)) + 1)
    print("reviewers "
          + ", ".join(f"{label or 'r%d' % i}=p{p:g}:c{c:g}"
                      for i, (label, p, c) in enumerate(args.reviewer, 1))
          + (" + combiner p%g:c%g" % args.combiner[1:] if args.combiner
             else ", no combiner")
          + f"; {args.attempts} attempts/cycle, {cycles} cycles "
          + f"({args.error_backoff} backoff) in {args.horizon_days} days")
    print(f"{'retries':>7}  {'P(attempt)':>10}  {'E[attempt]':>10}  "
          f"{'P(error)':>10}  {'E[cycle]':>10}  {'P(horizon)':>10}  "
          f"{'E[total]':>10}  {'dcost/dP':>10}")
    previous = None
    for retries in range(args.max_retries + 1):
        fails, cost = attempt_outcome(args.reviewer, args.combiner, retries)
        stalls, cycle_cost = retried_outcome(fails, cost, args.attempts)
        unreviewed, total = retried_outcome(fails, cost,
                                            args.attempts * cycles)
        if previous is None:
            marginal = f"{'-':>10}"
        else:
            removed = previous[0] - stalls
            marginal = (f"{(total - previous[1]) / removed:10.4g}"
                        if removed else f"{'inf':>10}")
        current = "  <- current budget" if retries == 1 else ""
        print(f"{retries:>7}  {fails:>10.4g}  {cost:>10.4g}  "
              f"{stalls:>10.4g}  {cycle_cost:>10.4g}  {unreviewed:>10.4g}  "
              f"{total:>10.4g}  {marginal}{current}")
        previous = (stalls, total)


if __name__ == "__main__":
    main()
