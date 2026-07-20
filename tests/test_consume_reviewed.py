"""consume_reviewed: the shared PR/issue manual-mode driver.

Pins the defer/retry pending-count compensation (a drift here blocks the
loop forever on ``reviewed_queue.get()``) and the apply/skip/quit
dispatch, for both the prompt_manual path and a ReviewUI.
"""
import unittest
from datetime import datetime, timezone
from queue import SimpleQueue
from threading import Thread
from unittest import mock

import fairy


def make_decision(n: int, action: str = "comment") -> fairy.Decision:
    return fairy.Decision(n, "t", "a", "-", action, "llm", None, "reply", "msg")


class Prepared:
    """Stands in for PreparedPR/PreparedIssue: anything not a Decision."""

    def __init__(self, n: int) -> None:
        self.number = n
        self.url = f"https://forge/pr/{n}"


class ConsumeReviewedTests(unittest.TestCase):
    def _consume(self, items, answers, *, echo_retries=False, cancelled=None,
                 on_choice=None):
        reviewed: SimpleQueue = SimpleQueue()
        llm: SimpleQueue = SimpleQueue()
        for item in items:
            reviewed.put(item)
        pending = fairy.PendingCount(len(items))
        applied: list[fairy.Decision] = []
        if echo_retries:
            # fake LLM worker: a retried item comes back re-reviewed
            def worker() -> None:
                p = llm.get()
                reviewed.put((p, make_decision(p.number)))
            Thread(target=worker, daemon=True).start()
        with mock.patch.object(fairy, "prompt_manual", side_effect=answers) as pm:
            decisions, stopped = fairy.consume_reviewed(
                reviewed, llm, pending,
                now=datetime.now(timezone.utc),
                manual=True, approve=False, kind="PR",
                apply=lambda p, d: applied.append(d),
                item_url=lambda p: p.url,
                cancelled=cancelled,
                on_choice=on_choice,
            )
        return decisions, stopped, applied, pending, pm

    def test_defer_then_retry_then_apply_terminates(self) -> None:
        p = Prepared(7)
        decisions, stopped, applied, pending, pm = self._consume(
            [(p, make_decision(7))], ["defer", "retry", "apply"],
            echo_retries=True,
        )
        self.assertEqual([d.pr_number for d in decisions], [7])
        self.assertEqual(len(applied), 1)
        self.assertFalse(stopped)
        self.assertEqual(pending.value, 0)
        self.assertEqual(pm.call_count, 3)
        self.assertIn("https://forge/pr/7", pm.call_args.kwargs["pr_url"])

    def test_hold_records_without_applying_and_polls_actions(self) -> None:
        # A "hold" ui (fairy-ui table mode) never blocks the consumer:
        # decisions are recorded, nothing is applied, and the action
        # channel is polled while draining.
        class HoldUI:
            def decide(self, prepared, d, url):
                return "hold"

            def item_done(self, prepared, d):
                pass

            def keep_open(self):
                return False

            def stopped(self):
                return False

        reviewed: SimpleQueue = SimpleQueue()
        llm: SimpleQueue = SimpleQueue()
        items = [(Prepared(1), make_decision(1)), (Prepared(2), make_decision(2))]
        for item in items:
            reviewed.put(item)
        applied: list[fairy.Decision] = []
        polls: list[int] = []
        decisions, stopped = fairy.consume_reviewed(
            reviewed, llm, fairy.PendingCount(2),
            now=datetime.now(timezone.utc),
            manual=False, approve=False, kind="PR",
            apply=lambda p, d: applied.append(d),
            item_url=lambda p: p.url,
            ui=HoldUI(),
            poll_actions=lambda: polls.append(1),
        )
        self.assertEqual([d.pr_number for d in decisions], [1, 2])
        self.assertEqual(applied, [])
        self.assertFalse(stopped)
        self.assertTrue(polls)

    def test_on_choice_reports_retry_skip_and_cancel(self) -> None:
        choices: list[tuple[int, str]] = []
        record = lambda d, c: choices.append((d.pr_number, c))  # noqa: E731
        self._consume([(Prepared(7), make_decision(7))], ["retry", "skip"],
                      echo_retries=True, on_choice=record)
        self._consume([(Prepared(6), make_decision(6))], [], cancelled={6},
                      on_choice=record)
        self.assertEqual(choices, [(7, "retry"), (7, "skip"), (6, "cancel")])

    def test_skip_records_without_applying(self) -> None:
        decisions, stopped, applied, pending, _ = self._consume(
            [(Prepared(1), make_decision(1))], ["skip"])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(applied, [])
        self.assertFalse(stopped)

    def test_quit_stops_before_remaining_items(self) -> None:
        items = [(Prepared(1), make_decision(1)), (Prepared(2), make_decision(2))]
        decisions, stopped, applied, pending, _ = self._consume(items, ["quit"])
        self.assertEqual([d.pr_number for d in decisions], [1])
        self.assertTrue(stopped)
        self.assertEqual(applied, [])

    def test_gate_skip_never_prompts(self) -> None:
        d = make_decision(3, action="skip")
        decisions, stopped, applied, pending, pm = self._consume([(d, d)], [])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(pm.call_count, 0)

    def test_retry_on_gate_decision_drops_item(self) -> None:
        d = make_decision(4)
        decisions, stopped, applied, pending, _ = self._consume([(d, d)], ["retry"])
        self.assertEqual(decisions, [])
        self.assertEqual(pending.value, 0)

    def test_cancelled_item_is_recorded_without_prompting(self) -> None:
        decisions, stopped, applied, pending, pm = self._consume(
            [(Prepared(6), make_decision(6))], [], cancelled={6})
        self.assertEqual([d.pr_number for d in decisions], [6])
        self.assertEqual(pm.call_count, 0)
        self.assertEqual(applied, [])
        self.assertFalse(stopped)

    def test_ui_idle_state_is_logged_once(self) -> None:
        class UI:
            def __init__(self) -> None:
                self.keeps = iter((True, True, False))

            def keep_open(self):
                return next(self.keeps)

            def stopped(self):
                return False

        with self.assertLogs(fairy.logger, level="INFO") as logs:
            decisions, stopped = fairy.consume_reviewed(
                SimpleQueue(), SimpleQueue(), fairy.PendingCount(0),
                now=datetime.now(timezone.utc),
                manual=False, approve=False, kind="PR",
                apply=lambda p, d: None, item_url=lambda p: "",
                ui=UI(),
            )
        self.assertEqual(decisions, [])
        self.assertEqual(
            sum("idle" in line for line in logs.output), 1)

    def test_review_ui_drives_decisions(self) -> None:
        class UI:
            def __init__(self) -> None:
                self.decided: list[tuple[int, str]] = []
                self.done: list[int] = []

            def decide(self, prepared, decision, url):
                self.decided.append((decision.pr_number, url))
                return "apply"

            def item_done(self, prepared, decision):
                self.done.append(decision.pr_number)

            def keep_open(self):
                return False

            def stopped(self):
                return False

        ui = UI()
        p = Prepared(9)
        reviewed: SimpleQueue = SimpleQueue()
        reviewed.put((p, make_decision(9)))
        applied: list[fairy.Decision] = []
        decisions, stopped = fairy.consume_reviewed(
            reviewed, SimpleQueue(), fairy.PendingCount(1),
            now=datetime.now(timezone.utc),
            manual=False, approve=False, kind="PR",
            apply=lambda pr, d: applied.append(d),
            item_url=lambda pr: pr.url,
            ui=ui,
        )
        self.assertEqual(ui.decided, [(9, "https://forge/pr/9")])
        self.assertEqual(ui.done, [9])
        self.assertEqual(len(applied), 1)
        self.assertFalse(stopped)


if __name__ == "__main__":
    unittest.main()
