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

--scan-closed-days: snapshot-only visibility for recently closed
tickets. Closed items must never reach the gates or the review queue
through this option, and their discussion cache is exempt from the
edit-catching TTL."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import agent  # noqa: E402

import forge_gcli  # noqa: E402
import issue_fairy  # noqa: E402

from test_agent import AgentCase, NOW, make_pr, verdict_ticket  # noqa: E402


def entry(n: int, updated: str, state: str = "closed") -> dict:
    return {"number": n, "updated_at": updated, "state": state}


class ListSinceTests(unittest.TestCase):
    """Paging must not trust the served page length: GitHub ignores
    ``limit`` (its parameter is ``per_page``, default 30 items,
    https://docs.github.com/en/rest/pulls/pulls) and Forgejo/Gitea cap
    ``limit`` at the server's MAX_RESPONSE_ITEMS (50 by default,
    https://forgejo.org/docs/latest/admin/config-cheat-sheet/), so a
    "short" page is normal and only an empty page or the cutoff ends
    the walk."""

    CUTOFF = datetime(2026, 7, 20, tzinfo=timezone.utc)

    def test_pages_stop_at_the_first_entry_past_the_cutoff(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="gitea")
        page1 = [entry(n, "2026-07-25T00:00:00Z") for n in range(50)]
        page2 = [entry(50, "2026-07-21T00:00:00Z"),
                 entry(51, "2026-07-19T00:00:00Z"),
                 entry(52, "2026-07-18T00:00:00Z")]
        with mock.patch.object(forge_gcli, "gcli_api",
                               side_effect=[page1, page2]) as api:
            got = forge_gcli.list_since(ns, "pulls", self.CUTOFF, state="closed")
        self.assertEqual([i["number"] for i in got], list(range(51)))
        self.assertEqual(api.call_count, 2)
        first = api.call_args_list[0].args[1]
        self.assertIn("state=closed", first)
        self.assertIn("sort=recentupdate", first)
        self.assertIn("limit=50", first)
        self.assertIn("page=1", first)

    def test_github_spells_sort_and_page_size_differently(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="github")
        with mock.patch.object(forge_gcli, "gcli_api",
                               return_value=[]) as api:
            forge_gcli.list_since(ns, "issues", self.CUTOFF, state="closed")
        path = api.call_args.args[1]
        self.assertIn("sort=updated", path)
        self.assertIn("direction=desc", path)
        self.assertIn("per_page=100", path)

    def test_a_short_page_does_not_end_the_listing(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="gitea")
        short = [entry(n, "2026-07-25T00:00:00Z") for n in range(30)]
        with mock.patch.object(forge_gcli, "gcli_api",
                               side_effect=[short, []]) as api:
            got = forge_gcli.list_since(ns, "pulls", self.CUTOFF, state="closed")
        self.assertEqual(len(got), 30)
        self.assertEqual(api.call_count, 2)

    def test_a_bad_updated_at_is_skipped_not_a_terminator(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="gitea")
        page = [entry(1, "2026-07-25T00:00:00Z"),
                {"number": 2, "updated_at": None},
                entry(3, "2026-07-24T00:00:00Z"),
                entry(4, "2026-07-01T00:00:00Z")]
        with mock.patch.object(forge_gcli, "gcli_api",
                               return_value=page):
            got = forge_gcli.list_since(ns, "pulls", self.CUTOFF, state="closed")
        self.assertEqual([i["number"] for i in got], [1, 3])


class NewestUpdatedAtTests(unittest.TestCase):
    def test_one_entry_of_any_state_most_recent_first(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="gitea")
        with mock.patch.object(forge_gcli, "gcli_api",
                               return_value=[entry(7, "2026-07-25T00:00:00Z")]) as api:
            got = forge_gcli.newest_updated_at(ns, "pr")
        self.assertEqual(got, "2026-07-25T00:00:00Z")
        path = api.call_args.args[1]
        self.assertIn("/pulls?", path)
        self.assertIn("state=all", path)
        self.assertIn("sort=recentupdate", path)
        self.assertIn("limit=1", path)

    def test_issues_on_github_and_an_empty_repo(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="github")
        with mock.patch.object(forge_gcli, "gcli_api",
                               return_value=[]) as api:
            self.assertIsNone(forge_gcli.newest_updated_at(ns, "issue"))
        path = api.call_args.args[1]
        self.assertIn("/issues?", path)
        self.assertIn("per_page=1", path)
        self.assertIn("direction=desc", path)




class IssuesSinceTests(unittest.TestCase):
    CUTOFF = datetime(2026, 7, 20, tzinfo=timezone.utc)

    def test_a_type_filter_where_the_forge_takes_one(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="gitea")
        with mock.patch.object(forge_gcli, "gcli_api", return_value=[]) as api:
            issue_fairy.list_issues_since(ns, self.CUTOFF)
        self.assertEqual(api.call_count, 1)
        self.assertIn("type=issues", api.call_args.args[1])
        self.assertIn("state=all", api.call_args.args[1])

    def test_pr_rows_are_subtracted_where_it_takes_none(self) -> None:
        ns = SimpleNamespace(owner="o", repo="r", forge_type="github")
        with mock.patch.object(
                forge_gcli, "list_since",
                side_effect=lambda a, endpoint, c, **kw:
                [entry(1, "x"), entry(2, "x")] if endpoint == "issues"
                else [entry(2, "x")]) as since:
            got = issue_fairy.list_issues_since(ns, self.CUTOFF)
        self.assertEqual([i["number"] for i in got], [1])
        self.assertEqual({c.args[1] for c in since.call_args_list},
                         {"issues", "pulls"})


class ScanClosedAgentTests(AgentCase):
    """Visibility only: snapshots refresh, the review pipeline and the
    closure lifecycle behave exactly as if the option were off."""
    P1, P2, P3 = ("2026-07-26T00:00:00Z", "2026-07-26T01:00:00Z",
                  "2026-07-26T02:00:00Z")

    def scan_with_closed(self, closed: list[dict], prs: list[dict],
                         fetch=None, probe: str = P1) -> None:
        self.ns.scan_closed_days = 7.0
        self.scan(prs, fetch=fetch, changed=closed, probe=probe)

    def test_closed_items_snapshot_but_never_reach_the_gates(self) -> None:
        closed = {**make_pr(2), "state": "closed", "merged": True}
        self.scan_with_closed([closed], [{**make_pr(1), "state": "open"}])
        self.assertEqual(self.db.find("pr", "1"), "queued")
        self.assertIsNone(self.db.find("pr", "2"))
        self.assertEqual(self.db.get("items", "pr", "2")["state"], "merged")
        numbers = [c.args[1]["number"] for c in self.prepare.call_args_list]
        self.assertEqual(numbers, [1])

    def test_closed_snapshots_skip_the_edit_ttl(self) -> None:
        closed = {**make_pr(2), "state": "closed", "merged": True}
        with mock.patch.object(agent, "_put_snapshot") as snap:
            self.scan_with_closed([closed], [{**make_pr(1), "state": "open"}])
        ages = {c.args[4]["number"]: c.args[6] for c in snap.call_args_list}
        self.assertEqual(ages[2], timedelta.max)
        self.assertEqual(ages[1], timedelta(
            hours=self.ns.discussion_cache_max_age_hours))

    def test_merged_at_alone_marks_the_snapshot_merged(self) -> None:
        """GitHub's PR listing payload (pull-request-simple schema) has
        ``merged_at`` but no ``merged`` boolean --
        https://docs.github.com/en/rest/pulls/pulls#list-pull-requests --
        while Forgejo/Gitea carry both."""
        closed = {**make_pr(2), "state": "closed",
                  "merged_at": "2026-07-20T00:00:00Z"}
        self.scan_with_closed([closed], [])
        self.assertEqual(self.db.get("items", "pr", "2")["state"], "merged")

    def test_unchanged_closed_items_skip_the_snapshot_rebuild(self) -> None:
        """The stored snapshot's updated_at is the memory, so the skip
        survives a restart."""
        closed = {**make_pr(2), "state": "closed", "merged": True}
        self.scan_with_closed([closed], [], probe=self.P1)
        self.scan_with_closed([closed], [], probe=self.P2)
        self.assertEqual(self.thread.call_count, 1)
        self.scan_with_closed(
            [dict(closed, updated_at="2026-07-21T00:00:00Z")], [], probe=self.P3)
        self.assertEqual(self.thread.call_count, 2)

    def test_with_the_option_off_a_listed_closure_cancels_but_is_not_kept(self) -> None:
        self.db.push("reviewed", "pr", "2", verdict_ticket(2))
        closed = {**make_pr(2), "state": "closed", "merged": True}
        self.scan([], changed=[closed], probe=self.P1)
        self.assertEqual(self.db.find("pr", "2"), "cancelled")
        self.assertIsNone(self.db.get("items", "pr", "2"))

    def test_a_failed_snapshot_is_listed_again_by_the_full_pass(self) -> None:
        with mock.patch.object(agent, "_put_snapshot",
                               side_effect=lambda db, ns, kind, token, *a:
                               token != "3"):
            self.scan_with_closed([entry(2, "2026-07-25T00:00:00Z"),
                                   entry(3, "2026-07-23T00:00:00Z"),
                                   entry(4, "2026-07-22T00:00:00Z")], [])
        self.assertEqual(self.db.read(self.db.root / agent.SCAN_STAMPS_DOC),
                         {"pr": "2026-07-25T00:00:00Z"})
        self.assertEqual(self.db.read(self.db.root / agent.SNAPSHOT_RETRY_DOC),
                         {"pr": "2026-07-23T00:00:00Z"})
        self.scan([], full=False)
        self.assertEqual(self.since.call_args.args[2],
                         datetime(2026, 7, 24, 23, 55, tzinfo=timezone.utc))
        self.scan_with_closed([], [])
        self.assertEqual(self.since.call_args.args[2],
                         datetime(2026, 7, 22, 23, 55, tzinfo=timezone.utc))
        self.assertEqual(self.db.read(self.db.root / agent.SNAPSHOT_RETRY_DOC), {})

    def test_closure_still_cancels_a_scanned_closed_tickets_verdict(self) -> None:
        self.db.push("reviewed", "pr", "2", verdict_ticket(2))
        closed = {**make_pr(2), "state": "closed", "merged": True}
        self.scan_with_closed([closed], [], fetch=lambda ns, n: closed)
        self.assertEqual(self.db.find("pr", "2"), "cancelled")
        self.assertEqual(self.db.get("cancelled", "pr", "2")["reason"],
                         "merged")


if __name__ == "__main__":
    unittest.main()
