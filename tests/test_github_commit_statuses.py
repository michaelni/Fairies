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

GitHub CI reaches fairy through the Checks API, not commit statuses.

GitHub Actions posts nothing to ``/commits/{sha}/statuses``; it reports
through ``/commits/{sha}/check-runs``. The fixtures are unmodified
``gcli -t github api`` captures from the project's own scratch
repository, whose ``.github/workflows/fixtures.yml`` exists to emit each
conclusion on demand -- a job that passes, one gated off with ``if:
false``, one that fails, and a slow one that can be caught still running
or cancelled by the next push:

* ``testrepo_pr4_statuses.json`` and ``testrepo_pr4_check_runs.json``
  are the two endpoints for the SAME commit -- 0 rows against 8 runs.
  That pair is the evidence for reading both: a status-only reader sees
  an Actions repo as having no CI at all and skips every PR.
* ``testrepo_check_runs_running.json`` was taken mid-flight, so it holds
  a run whose ``conclusion`` is still null.
* ``testrepo_check_runs_cancelled.json`` is the same commit after the
  next push cancelled that run.

Forgejo keeps the single request it always made; that is asserted here
rather than left to reviewers, because an extra round-trip per PR is a
regression fairy would pay on every run.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import forge_gcli  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "github"


def _load(name: str):
    with (FIXTURES / name).open() as f:
        return json.load(f)


def _args(forge_type: str) -> SimpleNamespace:
    return SimpleNamespace(forge_type=forge_type, gcli_account="", verbose=0)


def _statuses_for(forge_type: str, responses: list):
    """Run ``list_commit_statuses`` against scripted gcli responses.

    Returns ``(rows, requested_paths)``.
    """
    paths: list[str] = []

    def fake_api(args, path, **kwargs):
        paths.append(path)
        return responses[len(paths) - 1]

    with mock.patch.object(forge_gcli, "gcli_api", fake_api):
        rows = forge_gcli.list_commit_statuses(_args(forge_type), "o", "r", "sha")
    return rows, paths


def _states(fixture: str) -> dict[str, int]:
    rows, _ = _statuses_for("github", [[], _load(fixture)])
    counted: dict[str, int] = {}
    for row in rows:
        state = fairy.row_effective_state(row)
        counted[state] = counted.get(state, 0) + 1
    return counted


class EndpointSelectionTests(unittest.TestCase):

    def test_forgejo_asks_only_the_status_endpoint(self) -> None:
        rows, paths = _statuses_for("gitea", [[]])
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("/statuses"))
        self.assertEqual(rows, [])

    def test_github_asks_both_endpoints(self) -> None:
        _, paths = _statuses_for(
            "github", [_load("testrepo_pr4_statuses.json"),
                       _load("testrepo_pr4_check_runs.json")],
        )
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[0].endswith("/statuses"))
        self.assertTrue(paths[1].endswith("/check-runs"))

    def test_actions_ci_is_invisible_through_statuses_alone(self) -> None:
        self.assertEqual(_load("testrepo_pr4_statuses.json"), [])
        self.assertEqual(
            len(_load("testrepo_pr4_check_runs.json")["check_runs"]), 8,
        )

    def test_github_surfaces_the_runs_a_status_read_would_miss(self) -> None:
        rows, _ = _statuses_for(
            "github", [_load("testrepo_pr4_statuses.json"),
                       _load("testrepo_pr4_check_runs.json")],
        )
        self.assertEqual(len(rows), 8)
        self.assertIn("green", [r["context"] for r in rows])


class CheckRunProjectionTests(unittest.TestCase):

    def test_every_conclusion_reaches_the_state_fairy_acts_on(self) -> None:
        self.assertEqual(
            _states("testrepo_check_runs_running.json"),
            {"SUCCESS": 1, "NEUTRAL": 1, "FAILURE": 1, "PENDING": 1},
        )

    def test_a_cancelled_run_is_its_own_state(self) -> None:
        self.assertEqual(
            _states("testrepo_check_runs_cancelled.json").get("CANCELLED"), 1)

    def test_row_carries_the_keys_the_contract_promises(self) -> None:
        rows, _ = _statuses_for("github", [[], _load("testrepo_pr4_check_runs.json")])
        self.assertEqual(
            sorted(rows[0]),
            ["context", "created_at", "description", "state", "target_url",
             "updated_at"],
        )
        self.assertTrue(rows[0]["target_url"].startswith("https://github.com/"))

    def test_a_failing_run_reaches_the_triage_payload_with_its_link(self) -> None:
        rows, _ = _statuses_for(
            "github", [[], _load("testrepo_check_runs_running.json")])
        details = fairy.build_ci_failure_details(rows)
        self.assertEqual([d["context"] for d in details], ["red"])
        self.assertTrue(details[0]["target_url"].startswith("https://github.com/"))


class SkippedChecksAreNotFailuresTests(unittest.TestCase):
    """A conditional job that did not run must not skip the whole PR.

    ``never-runs`` is gated off and reports ``skipped``, which maps to
    NEUTRAL. prepare_pr's failing set was "anything not SUCCESS", so a PR
    carrying one took the "CI not successful" skip with an empty triage
    payload -- no LLM, no explanation. Conditional and path-filtered jobs
    are the norm on Actions, so that was every GitHub PR.
    """

    def test_the_capture_really_does_contain_a_skipped_check(self) -> None:
        self.assertEqual(_states("testrepo_check_runs_running.json")["NEUTRAL"], 1)

    def test_a_skipped_check_alone_leaves_nothing_failing(self) -> None:
        rows, _ = _statuses_for(
            "github", [[], _load("testrepo_check_runs_running.json")])
        skipped = [r for r in rows if fairy.row_effective_state(r) == "NEUTRAL"]
        failing = sorted(
            ctx for ctx, (state, _) in
            fairy.effective_commit_statuses(skipped).items()
            if state not in ("SUCCESS", "NEUTRAL")
        )
        self.assertEqual(failing, [])

    def test_a_real_failure_is_still_failing(self) -> None:
        rows, _ = _statuses_for(
            "github", [[], _load("testrepo_check_runs_running.json")])
        failing = sorted(
            ctx for ctx, (state, _) in
            fairy.effective_commit_statuses(rows).items()
            if state not in ("SUCCESS", "NEUTRAL")
        )
        self.assertEqual(failing, ["red", "slow"])


class ForgejoRowsAreUnaffectedTests(unittest.TestCase):
    """The Forgejo spelling still lands on the same keys (both shapes pinned)."""

    def test_forgejo_status_row_projects_to_the_same_keys(self) -> None:
        rows, _ = _statuses_for("gitea", [[{
            "context": "/ build", "status": "failure",
            "description": "Has been cancelled",
            "target_url": "https://forge/actions/runs/1/jobs/0",
            "created_at": "2026-04-26T20:00:00Z",
            "updated_at": "2026-04-26T20:00:00Z",
        }]])
        self.assertEqual(rows[0]["state"], "failure")
        self.assertEqual(fairy.row_effective_state(rows[0]), "CANCELLED")


if __name__ == "__main__":
    unittest.main()
