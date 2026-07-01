"""Tests for the LLM skip-backoff cache.

When the LLM classifies a PR as ``"skip"`` it leaves no on-PR trace,
so without a backoff every subsequent run would re-spend an LLM call
on the same unchanged PR forever. The backoff cache fixes that with
strict doubling (24h, 48h, 96h, ...) and never permanently caches:
any push or new discussion activity bypasses it, and any non-skip
verdict resets the counter.

These tests pin the three pure helpers (``backoff_for_consecutive_skips``,
``compute_llm_skip_backoff``, ``writeback_llm_skip_backoff``) so the
gate semantics are readable in one place. Integration with the rest
of ``prepare_pr`` is covered structurally by the existing prepare-pr
tests; the gate is a single early-return that depends only on these
three helpers.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bot_state  # noqa: E402
import fairy  # noqa: E402


T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
HEAD = "deadbeefcafe"
LAST_ACT = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)
LAST_ACT_ISO = LAST_ACT.isoformat()


def _skip_entry(
    *,
    consec: int,
    last_at: datetime,
    head: str | None = HEAD,
    last_act_iso: str | None = LAST_ACT_ISO,
) -> dict[str, object]:
    return {
        "last_llm_decision": "skip",
        "last_llm_at": last_at.isoformat(),
        "last_llm_head_sha": head,
        "last_llm_last_activity_iso": last_act_iso,
        "consecutive_skip_count": consec,
    }


class BackoffMathTests(unittest.TestCase):
    """``backoff_for_consecutive_skips`` is the user-facing schedule:
    24h after the 1st skip, 48h after the 2nd, 96h after the 3rd, ...
    No policy cap; only a structural overflow guard at exponent=25.
    """

    def test_zero_or_negative_count_is_no_backoff(self) -> None:
        for n in (-1, 0):
            self.assertEqual(
                fairy.backoff_for_consecutive_skips(n),
                timedelta(0),
                f"n={n}",
            )

    def test_doubling_schedule(self) -> None:
        cases = [
            (1, 24),
            (2, 48),
            (3, 96),
            (4, 192),
            (5, 384),
        ]
        for n, hours in cases:
            self.assertEqual(
                fairy.backoff_for_consecutive_skips(n),
                timedelta(hours=hours),
                f"n={n}",
            )

    def test_overflow_guard_clamps_exponent_at_25(self) -> None:
        # n=26 and n=1000 must produce the SAME timedelta (the clamp).
        # Any divergence here means the clamp is broken or moved.
        capped = fairy.backoff_for_consecutive_skips(26)
        self.assertEqual(
            capped, timedelta(hours=24 * (2 ** 25)),
        )
        self.assertEqual(
            fairy.backoff_for_consecutive_skips(1000), capped,
        )


class ComputeLlmSkipBackoffTests(unittest.TestCase):
    """``compute_llm_skip_backoff`` returns ``(consec, eligible_at)`` to
    suppress, or ``None`` to let the LLM run. Pure function, no I/O.
    """

    def test_empty_entry_does_not_suppress(self) -> None:
        self.assertIsNone(
            fairy.compute_llm_skip_backoff({}, HEAD, LAST_ACT, T0),
        )

    def test_non_skip_last_decision_does_not_suppress(self) -> None:
        entry = _skip_entry(consec=1, last_at=T0)
        entry["last_llm_decision"] = "ok_approve"
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(entry, HEAD, LAST_ACT, T0),
        )

    def test_within_window_suppresses(self) -> None:
        entry = _skip_entry(consec=1, last_at=T0 - timedelta(hours=1))
        # 24h window after 1 skip; only 1h elapsed => suppress.
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, LAST_ACT, T0,
        )
        self.assertIsNotNone(result)
        assert result is not None
        consec, eligible = result
        self.assertEqual(consec, 1)
        self.assertEqual(eligible, T0 - timedelta(hours=1) + timedelta(hours=24))

    def test_after_window_does_not_suppress(self) -> None:
        # 24h elapsed exactly at the boundary => caller may run LLM again.
        entry = _skip_entry(consec=1, last_at=T0 - timedelta(hours=24))
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(entry, HEAD, LAST_ACT, T0),
        )

    def test_doubled_window_after_two_skips(self) -> None:
        # After 2 skips the window is 48h. Only 30h elapsed => still suppress.
        entry = _skip_entry(consec=2, last_at=T0 - timedelta(hours=30))
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, LAST_ACT, T0,
        )
        self.assertIsNotNone(result)
        # And once 48h elapses we're free.
        entry_old = _skip_entry(consec=2, last_at=T0 - timedelta(hours=48))
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(
                entry_old, HEAD, LAST_ACT, T0,
            ),
        )

    def test_head_sha_change_bypasses_window(self) -> None:
        # A push (different head_sha) must let the LLM re-run even
        # mid-window. This is one of the two cache-bypass mechanisms.
        entry = _skip_entry(consec=5, last_at=T0 - timedelta(hours=1))
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(
                entry, "newhead123", LAST_ACT, T0,
            ),
        )

    def test_last_activity_change_bypasses_window(self) -> None:
        # A new comment / review (different last_activity) must let the
        # LLM re-run mid-window. This is the second bypass mechanism --
        # crucial for @-mention forced reviews to keep working.
        entry = _skip_entry(consec=5, last_at=T0 - timedelta(hours=1))
        new_activity = LAST_ACT + timedelta(minutes=1)
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(
                entry, HEAD, new_activity, T0,
            ),
        )

    def test_zero_counter_does_not_suppress(self) -> None:
        # Defensive: a "skip" decision with consec=0 must not suppress.
        # (Shouldn't happen via writeback, but old / corrupted cache
        # entries from a different code version must not lock the bot.)
        entry = _skip_entry(consec=0, last_at=T0 - timedelta(hours=1))
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(entry, HEAD, LAST_ACT, T0),
        )

    def test_corrupt_last_at_does_not_suppress(self) -> None:
        entry = _skip_entry(consec=2, last_at=T0)
        entry["last_llm_at"] = "not-a-date"
        self.assertIsNone(
            fairy.compute_llm_skip_backoff(entry, HEAD, LAST_ACT, T0),
        )

    def test_none_last_activity_matched_by_none(self) -> None:
        # PRs without resolvable last_activity are still cacheable so
        # long as both writeback and gate agree the value is None.
        entry = _skip_entry(
            consec=1, last_at=T0 - timedelta(hours=1), last_act_iso=None,
        )
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, None, T0,
        )
        self.assertIsNotNone(result)


class WritebackLlmSkipBackoffTests(unittest.TestCase):
    """``writeback_llm_skip_backoff`` is the single funnel after every
    LLM call. ``"skip"`` increments the counter, anything else resets,
    transient errors write nothing.
    """

    KEY = bot_state.Key("o", "r", 42)

    def _writeback(self, state: bot_state.State, classification: str) -> None:
        fairy.writeback_llm_skip_backoff(
            state, "o", "r", 42, HEAD, LAST_ACT, classification,
        )

    def test_skip_initializes_counter_to_one(self) -> None:
        state = bot_state.State()
        self._writeback(state, "skip")
        entry = state.entries[self.KEY]
        self.assertEqual(entry["last_llm_decision"], "skip")
        self.assertEqual(entry["last_llm_head_sha"], HEAD)
        self.assertEqual(entry["last_llm_last_activity_iso"], LAST_ACT_ISO)
        self.assertEqual(entry["consecutive_skip_count"], 1)
        self.assertIsInstance(entry["last_llm_at"], str)

    def test_repeated_skip_increments_counter(self) -> None:
        state = bot_state.State()
        state.entries[self.KEY] = {"consecutive_skip_count": 3}
        self._writeback(state, "skip")
        self.assertEqual(state.entries[self.KEY]["consecutive_skip_count"], 4)

    def test_non_skip_resets_counter(self) -> None:
        state = bot_state.State()
        state.entries[self.KEY] = {
            "last_llm_decision": "skip",
            "consecutive_skip_count": 7,
        }
        self._writeback(state, "ok_approve")
        entry = state.entries[self.KEY]
        self.assertEqual(entry["last_llm_decision"], "ok_approve")
        self.assertEqual(entry["consecutive_skip_count"], 0)

    def test_error_classification_writes_nothing(self) -> None:
        # A transient LLM/network error must NOT poison the counter
        # nor shift the next-eligible timestamp.
        original: bot_state.Entry = {
            "last_llm_decision": "skip",
            "last_llm_at": "2026-04-01T00:00:00+00:00",
            "last_llm_head_sha": "oldhead",
            "last_llm_last_activity_iso": "2026-04-01T00:00:00+00:00",
            "consecutive_skip_count": 2,
        }
        state = bot_state.State()
        state.entries[self.KEY] = dict(original)
        self._writeback(state, "error")
        self.assertEqual(state.entries[self.KEY], original)

    def test_placeholder_classification_writes_nothing(self) -> None:
        # ``Decision`` defaults ``llm_classification`` to "-" when no
        # LLM verdict is present (e.g. wrapper crash before the call).
        # That placeholder must also be a no-op.
        state = bot_state.State()
        self._writeback(state, "-")
        self.assertNotIn(self.KEY, state.entries)

    def test_roundtrip_skip_then_compute_suppresses(self) -> None:
        # End-to-end: writeback a skip, immediately compute the gate;
        # the resulting state must suppress within the 24h window.
        state = bot_state.State()
        self._writeback(state, "skip")
        entry = state.entries[self.KEY]
        # Fast-forward 1h: still within 24h window.
        last_at = fairy.iso_to_dt(entry["last_llm_at"])
        assert last_at is not None
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, LAST_ACT, last_at + timedelta(hours=1),
        )
        self.assertIsNotNone(result)
        # Fast-forward 25h: window has elapsed.
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, LAST_ACT, last_at + timedelta(hours=25),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
