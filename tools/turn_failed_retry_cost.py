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
``ProviderTurnFailed`` in llm_review_api.py); ``run_parallel`` and
``review_pr`` retry the blocked stage, today with a budget of one
retry. For each retry budget up to --max-retries this prints the
probability that a run of parallel reviewers plus combiner still fails
on the block, the expected cost of the run, cost per unit of failure
probability, and the marginal cost of the probability removed by that
budget's last retry.

Model (mirrors run_parallel / review_pr):
  * each attempt is blocked independently with probability -p;
  * a reviewer that exhausts the budget is dropped; the run fails only
    when every reviewer is dropped or the combiner exhausts the budget;
  * the combiner runs (and costs) only when a draft survives;
  * every attempt, blocked or not, is billed one full stage cost;
  * failure causes other than the block are out of scope, as is the
    outer --llm-max-attempts whole-pipeline retry.

Costs default to 1, so without --reviewer-cost / --combiner-cost the
cost columns are in units of one stage attempt.

Usage:
  tools/turn_failed_retry_cost.py -p 0.3
  tools/turn_failed_retry_cost.py -p 0.05 --reviewers 3 \\
      --reviewer-cost 0.90 --combiner-cost 1.40 --max-retries 6
"""

from __future__ import annotations

import argparse


def failure_probability(p: float, reviewers: int, retries: int) -> float:
    """P that the run fails on the block: every reviewer dropped, or the
    combiner blocked on all retries+1 attempts. All args as in main()."""
    blocked = p ** (retries + 1)
    all_dropped = blocked ** reviewers
    return all_dropped + (1 - all_dropped) * blocked


def expected_cost(p: float, reviewers: int, retries: int,
                  reviewer_cost: float, combiner_cost: float) -> float:
    """Expected cost of one run: every stage pays per attempt until it
    succeeds or the budget is spent. All args as in main()."""
    blocked = p ** (retries + 1)
    attempts = sum(p ** i for i in range(retries + 1))
    return attempts * (reviewers * reviewer_cost
                       + (1 - blocked ** reviewers) * combiner_cost)


def probability(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text}: not a probability in [0, 1]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cost vs probability of losing a review run to a "
                    "provider-ended turn, per retry budget")
    parser.add_argument("-p", "--block-probability", type=probability,
                        required=True,
                        help="per-attempt probability of the block, in [0, 1]")
    parser.add_argument("--reviewers", type=int, default=3,
                        help="parallel model reviewers (default: 3)")
    parser.add_argument("--reviewer-cost", type=float, default=1.0,
                        help="cost of one reviewer attempt (default: 1)")
    parser.add_argument("--combiner-cost", type=float, default=1.0,
                        help="cost of one combiner attempt (default: 1)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="largest retry budget to tabulate (default: 4)")
    args = parser.parse_args()

    print(f"block probability {args.block_probability:g} per attempt, "
          f"{args.reviewers} reviewer(s) + combiner")
    print(f"{'retries':>7}  {'P(run fails)':>12}  {'E[cost]':>10}  "
          f"{'E[cost]/P':>10}  {'dcost/dP':>10}")
    previous = None
    for retries in range(args.max_retries + 1):
        p_fail = failure_probability(
            args.block_probability, args.reviewers, retries)
        cost = expected_cost(
            args.block_probability, args.reviewers, retries,
            args.reviewer_cost, args.combiner_cost)
        per_p = f"{cost / p_fail:10.4g}" if p_fail else f"{'inf':>10}"
        if previous is None:
            marginal = f"{'-':>10}"
        else:
            removed = previous[0] - p_fail
            marginal = (f"{(cost - previous[1]) / removed:10.4g}"
                        if removed else f"{'inf':>10}")
        current = "  <- current budget" if retries == 1 else ""
        print(f"{retries:>7}  {p_fail:>12.4g}  {cost:>10.4g}  "
              f"{per_p}  {marginal}{current}")
        previous = (p_fail, cost)


if __name__ == "__main__":
    main()
