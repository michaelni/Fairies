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

worker: claims become verdict tickets in the right directories."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import filedb  # noqa: E402
import worker  # noqa: E402
import workset  # noqa: E402


def make_pr(n: int) -> dict:
    return {"number": n, "title": f"t{n}", "user": {"login": "a"},
            "updated_at": "2026-07-19T10:00:00Z", "head": {"sha": f"h{n}"},
            "html_url": f"https://forge/pr/{n}"}


def queued_ticket(n: int, backoff: float = 0.0) -> dict:
    prepared = fairy.PreparedPR(
        pr=make_pr(n), number=n, title=f"t{n}", author="a", auto_merge="-",
        last_activity=None, base_reason="review", discussion=[],
        reviewer_username="fairy")
    return {"title": f"t{n}", "author": "a", "skip_backoff_h": backoff,
            "forced": False, "prepared": fairy.prepared_to_dict(prepared)}


def decision(n: int, action: str = "comment", llm: str = "moderate_issues",
             msg: str = "m", labels: tuple = (),
             auto_merge: str = "-") -> fairy.Decision:
    return fairy.Decision(n, f"t{n}", "a", auto_merge, action, "llm", None,
                          llm, msg, label_changes=labels)


class WorkerCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))
        self.ns = fairy.parse_args(["--owner", "o", "--repo", "r"])

    def run_one(self, n: int, result) -> str:
        claim = self.db.claim("queued", "llm", "pr", str(n))
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=result):
            return worker.review_claim(claim, self.ns)


class VerdictRoutingTests(WorkerCase):
    def test_actionable_review_lands_in_reviewed(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5))
        state = self.run_one(5, lambda ns, p: decision(5))
        self.assertEqual(state, "reviewed")
        t = self.db.get("reviewed", "pr", "5")
        self.assertEqual(t["review"]["classification"], "moderate_issues")
        self.assertEqual(t["expected_updated_at"], "2026-07-19T10:00:00Z")
        self.assertEqual(t["expected_head_ref"], "h5")
        self.assertNotIn("prepared", t)  # the payload is spent
        self.assertIsNone(self.db.get("llm", "pr", "5"))

    def test_llm_skip_keeps_its_backoff_in_skipped(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5, backoff=48))
        state = self.run_one(5, lambda ns, p: decision(5, action="skip",
                                                       llm="skip", msg=""))
        self.assertEqual(state, "skipped")
        t = self.db.get("skipped", "pr", "5")
        self.assertEqual(t["skip_backoff_h"], 48)
        self.assertTrue(t["llm_at"])

    def test_auto_merge_state_survives_into_the_verdict(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5))
        self.run_one(5, lambda ns, p: decision(5, action="approve",
                                               auto_merge="merge"))
        t = self.db.get("reviewed", "pr", "5")
        self.assertEqual(t["auto_merge"], "merge")

    def test_skip_with_label_changes_is_operator_actionable(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5))
        labels = (fairy.LabelChange("needs docs", "add", "", False),)
        state = self.run_one(5, lambda ns, p: decision(5, action="skip",
                                                       llm="skip", labels=labels))
        self.assertEqual(state, "reviewed")

    def test_error_lands_in_error(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5))
        state = self.run_one(5, lambda ns, p: decision(5, action="error",
                                                       llm="error", msg=""))
        self.assertEqual(state, "error")
        t = self.db.get("error", "pr", "5")
        self.assertEqual(t["error"], "llm")
        # the guard makes an operator x on the error row stick
        self.assertEqual(t["expected_updated_at"], "2026-07-19T10:00:00Z")

    def test_wrapper_stage_notes_reach_the_claimed_ticket(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5))

        def fake_llm(ns, prepared):
            # the wrapper writes its progress through the override path
            workset.update_json(Path(ns.workset_file_override),
                                lambda d: d.__setitem__("stage", "review"))
            return decision(5)

        self.run_one(5, fake_llm)
        t = self.db.get("reviewed", "pr", "5")
        self.assertNotIn("stage", t)  # transient progress, spent with the run


class OperatorVetoTests(WorkerCase):
    def test_cancel_flag_discards_the_verdict(self) -> None:
        self.db.push("queued", "pr", "5", queued_ticket(5))

        def llm_with_midway_cancel(ns, prepared):
            workset.update_json(
                Path(ns.workset_file_override),
                lambda d: d.update(cancel=True, reason="operator cancel"))
            return decision(prepared.number)

        state = self.run_one(5, llm_with_midway_cancel)
        self.assertEqual(state, "cancelled")
        t = self.db.get("cancelled", "pr", "5")
        self.assertEqual(t["reason"], "operator cancel")
        self.assertNotIn("review", t)
        self.assertIsNone(self.db.get("llm", "pr", "5"))


class DrainTests(WorkerCase):
    def test_drain_reviews_every_queued_ticket_of_its_kinds(self) -> None:
        for n in (1, 2):
            self.db.push("queued", "pr", str(n), queued_ticket(n))
        self.db.push("queued", "issue", "3", {"title": "i"})  # no issue side
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=lambda ns, p: decision(p.number)):
            done = worker.drain(self.db, {"pr": self.ns})
        self.assertEqual(done, 2)
        self.assertEqual(self.db.list_state("reviewed"), [("pr", "1"), ("pr", "2")])
        self.assertEqual(self.db.list_state("queued"), [("issue", "3")])

    def test_parallel_drain_reviews_concurrently_with_isolated_ns(self) -> None:
        # Three tickets, three threads: each review must see its OWN
        # workset_file_override -- a shared namespace would send one
        # ticket's wrapper notes into another ticket's file.
        import threading
        for n in (1, 2, 3):
            self.db.push("queued", "pr", str(n), queued_ticket(n))
        gate = threading.Barrier(3, timeout=10)

        def fake_llm(ns, prepared):
            gate.wait()  # proves all three reviews really overlap
            workset.update_json(Path(ns.workset_file_override),
                                lambda d: d.__setitem__("seen", prepared.number))
            return decision(prepared.number)

        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            done = worker.drain(self.db, {"pr": self.ns}, parallel=3)
        self.assertEqual(done, 3)
        for n in (1, 2, 3):
            self.assertEqual(self.db.get("reviewed", "pr", str(n))["seen"], n)

    def test_a_slow_review_never_idles_the_other_slots(self) -> None:
        # a barrier round would wait for #1 before ever starting #3;
        # the top-up must run #3 (pushed mid-drain) while #1 still holds
        # its slot, or --parallel is parallel in name only
        import threading
        import time as _time
        release = threading.Event()

        def fake_llm(ns, prepared):
            if prepared.number == 1:
                self.assertTrue(release.wait(10), "top-up never happened")
            elif prepared.number == 2:
                self.db.push("queued", "pr", "3", queued_ticket(3))
            elif prepared.number == 3:
                release.set()
            return decision(prepared.number)

        for n in (1, 2):
            self.db.push("queued", "pr", str(n), queued_ticket(n))
        t0 = _time.monotonic()
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            done = worker.drain(self.db, {"pr": self.ns}, parallel=2)
        self.assertEqual(done, 3)
        self.assertLess(_time.monotonic() - t0, 5)

    def test_forced_tickets_are_claimed_first(self) -> None:
        order = []
        for n in (1, 2, 3):
            ticket = queued_ticket(n)
            ticket["forced"] = n == 3
            self.db.push("queued", "pr", str(n), ticket)

        def fake_llm(ns, prepared):
            order.append(prepared.number)
            return decision(prepared.number)

        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            worker.drain(self.db, {"pr": self.ns})
        self.assertEqual(order, [3, 1, 2])

    def test_arrivals_are_claimed_while_all_started_reviews_run(self) -> None:
        """Production 2026-07-28: 10 queued, 1 in llm/, 2 slots idle --
        wait(FIRST_COMPLETED) slept until the long review ended."""
        import threading
        release = threading.Event()

        def fake_llm(ns, prepared):
            if prepared.number == 1:
                self.db.push("queued", "pr", "2", queued_ticket(2))
                self.assertTrue(release.wait(10), "arrival never picked up")
            else:
                release.set()
            return decision(prepared.number)

        self.db.push("queued", "pr", "1", queued_ticket(1))
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=fake_llm):
            done = worker.drain(self.db, {"pr": self.ns}, parallel=2)
        self.assertEqual(done, 2)

    def test_broken_ticket_lands_in_error_and_does_not_starve(self) -> None:
        # A ticket the worker cannot even read must not return to
        # queued/: sorted first, it would be re-claimed on every pass
        # and the worker would never review anything again.
        self.db.push("queued", "pr", "1", {"title": "broken: no prepared"})
        self.db.push("queued", "pr", "2", queued_ticket(2))
        with mock.patch.object(fairy, "safe_apply_llm_review_to_prepared",
                               side_effect=lambda ns, p: decision(p.number)):
            done = worker.drain(self.db, {"pr": self.ns})
        self.assertEqual(done, 1)
        self.assertEqual(self.db.find("pr", "2"), "reviewed")
        t = self.db.get("error", "pr", "1")
        self.assertIn("prepared", t["error"])
        self.assertTrue(t["llm_at"])


class _StopLoop(BaseException):
    """Sentinel to end the worker loop; a BaseException so the loop's
    own ``except Exception`` cannot swallow it."""


class LoopTests(unittest.TestCase):
    """--loop N is the daemon contract, the same one the agent keeps:
    keep draining, and survive a failed drain. Without it the worker
    drains once and an error is fatal, so cron sees the exit code."""

    def run_main(self, flags: str, outcomes: list) -> list[int]:
        import shlex
        import time
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker.db_config.write_config(Path(tmp.name), "o/r", set(),
                                  {"owner": "o", "repo": "r"}, None)
        argv = ["worker.py", "--db-root", tmp.name] + shlex.split(flags)
        calls = self.calls = []

        def drain(*args, **kwargs) -> int:
            calls.append(len(calls))
            outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if outcome is not None:
                raise outcome
            return 0

        wake = mock.Mock()
        wake.wait.side_effect = lambda timeout=None: time.sleep(timeout or 0)
        with mock.patch.object(worker, "drain", side_effect=drain), \
                mock.patch.object(worker, "setup_logging"), \
                mock.patch.object(worker, "watch_paths"), \
                mock.patch.object(worker, "Event", return_value=wake), \
                mock.patch.object(sys, "argv", argv):
            self.rc = worker.main()
        return calls

    def test_loop_keeps_draining(self) -> None:
        with self.assertRaises(_StopLoop):
            self.run_main("--loop 0.01", [None, None, _StopLoop()])
        self.assertEqual(len(self.calls), 3)

    def test_without_loop_a_single_drain_returns(self) -> None:
        self.assertEqual(self.run_main("", [None]), [0])
        self.assertEqual(self.rc, 0)

    def test_a_failed_drain_does_not_kill_the_daemon(self) -> None:
        """A wrapper/provider outage costs one interval, not the whole
        service."""
        with self.assertRaises(_StopLoop):
            self.run_main("--loop 0.01", [RuntimeError("provider 503"),
                                          _StopLoop()])
        self.assertEqual(len(self.calls), 2)

    def test_a_failed_drain_is_fatal_in_one_shot_mode(self) -> None:
        with self.assertRaises(RuntimeError):
            self.run_main("", [RuntimeError("provider 503")])


class ConfigGuardTests(unittest.TestCase):
    def test_config_without_side_strings_is_a_clear_exit(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker.db_config.write_config(Path(tmp.name), "o/r", set(), None, None)
        with mock.patch.object(sys, "argv", ["worker.py", "--db-root", tmp.name]), \
                self.assertRaisesRegex(SystemExit, "no side options"):
            worker.main()

    def test_a_hand_broken_config_is_rejected_at_startup(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker.db_config.write_config(Path(tmp.name), "o/r", set(),
                                      {"owner": "o", "repo": "r",
                                       "llm-review-cmd": "wrapper"}, None)
        with mock.patch.object(worker, "setup_logging"), \
                mock.patch.object(worker, "drain"), \
                mock.patch.object(sys, "argv",
                                  ["worker.py", "--db-root", tmp.name]), \
                self.assertRaises(SystemExit) as ctx:
            worker.main()
        self.assertEqual(ctx.exception.code, 2)


class OverrideTests(unittest.TestCase):
    def drain_sides(self, argv: list, pr: dict | None = None,
                    issue: dict | None = None) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker.db_config.write_config(Path(tmp.name), "o/r", set(), pr, issue)
        captured: dict = {}

        def drain(db, sides, **kwargs) -> int:
            captured.update(sides)
            return 0

        with mock.patch.object(worker, "drain", side_effect=drain), \
                mock.patch.object(worker, "setup_logging"), \
                mock.patch.object(sys, "argv",
                                  ["worker.py", "--db-root", tmp.name] + argv):
            worker.main()
        return captured

    def test_help_lists_the_overridable_side_options(self) -> None:
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["worker.py", "--help"]), \
                contextlib.redirect_stdout(buf):
            rc = worker.main()
        self.assertEqual(rc, 0)
        self.assertIn("--llm-review-cmd", buf.getvalue())
        self.assertIn("--issue-label", buf.getvalue())
        self.assertIn("defaults from config.toml", buf.getvalue())
        self.assertNotIn("--min-age-days", buf.getvalue())

    def test_cli_overrides_replace_config_values(self) -> None:
        sides = self.drain_sides(
            ["--llm-review-cmd", "w2"],
            pr={"owner": "o", "repo": "r", "llm-review-cmd": "w1",
                "patch-repo": "p"})
        self.assertEqual(sides["pr"].llm_review_cmd, "w2")
        self.assertEqual(sides["pr"].patch_repo, Path("p"))

    def test_shared_overrides_hit_both_sides_and_sections_one(self) -> None:
        sides = self.drain_sides(
            ["--llm-timeout", "99", "--issues", "--llm-retry-delay", "9"],
            pr={"owner": "o", "repo": "r"},
            issue={"owner": "o", "repo": "r"})
        self.assertEqual(sides["pr"].llm_timeout, 99)
        self.assertEqual(sides["issue"].llm_timeout, 99)
        self.assertEqual(sides["issue"].llm_retry_delay, 9)
        self.assertEqual(sides["pr"].llm_retry_delay,
                         fairy.parse_args(["--owner", "o", "--repo", "r"])
                         .llm_retry_delay)

    def test_an_append_override_replaces_the_config_list(self) -> None:
        sides = self.drain_sides(
            ["--podman-host", "b"],
            pr={"owner": "o", "repo": "r", "podman-host": ["a"]})
        self.assertEqual(sides["pr"].podman_host, ["b"])

    def test_a_shared_override_of_a_one_side_option_routes_to_its_side(
            self) -> None:
        sides = self.drain_sides(
            ["--codex-host", "h"],
            pr={"owner": "o", "repo": "r"},
            issue={"owner": "o", "repo": "r"})
        self.assertEqual(sides["pr"].codex_host, "h")
        self.assertIn("issue", sides)

    def test_agent_scope_config_keys_are_ignored(self) -> None:
        sides = self.drain_sides([], pr={"owner": "o", "repo": "r",
                                         "min-age-days": "5"})
        self.assertFalse(hasattr(sides["pr"], "min_age_days"))

    def test_an_unknown_config_key_is_still_an_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.drain_sides([], pr={"owner": "o", "repo": "r",
                                     "no-such-option": "x"})
        self.assertEqual(ctx.exception.code, 2)


class ColorTests(unittest.TestCase):
    def test_wrapper_stream_logger_joins_the_side_log_file(self) -> None:
        """The pane tails the file; without forge_gcli's logger there the
        whole live wrapper stream is invisible during a review."""
        import forge_gcli
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker.db_config.write_config(
            Path(tmp.name), "o/r", set(),
            {"owner": "o", "repo": "r", "log-file": f"{tmp.name}/side.log"},
            None)
        argv = ["worker.py", "--db-root", tmp.name]
        with mock.patch.object(worker, "add_file_log") as file_log, \
                mock.patch.object(worker, "setup_logging"), \
                mock.patch.object(worker, "drain"), \
                mock.patch.object(sys, "argv", argv):
            worker.main()
        self.assertIn(forge_gcli.logger, file_log.call_args.args)

    def test_side_color_reaches_setup_logging(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker.db_config.write_config(Path(tmp.name), "o/r", set(),
                                  {"owner": "o", "repo": "r",
                                   "color": "always"}, None)
        argv = ["worker.py", "--db-root", tmp.name]
        with mock.patch.object(worker, "setup_logging") as logging_setup, \
                mock.patch.object(worker, "drain"), \
                mock.patch.object(sys, "argv", argv):
            worker.main()
        self.assertEqual(logging_setup.call_args.kwargs["color"], "always")


if __name__ == "__main__":
    unittest.main()
