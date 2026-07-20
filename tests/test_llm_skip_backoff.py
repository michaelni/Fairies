"""Tests for the LLM skip-backoff cache.

When the LLM classifies a PR as ``"skip"`` it leaves no on-PR trace,
so without a backoff every subsequent run would re-spend an LLM call
on the same unchanged PR forever. The backoff cache fixes that with
strict doubling (24h, 48h, 96h, ...) and never permanently caches:
any push or new discussion activity bypasses it, and any non-skip
verdict resets the counter.

These tests pin the pure helpers (``backoff_for_consecutive_skips``,
``compute_llm_skip_backoff``) and the workset-file bookkeeping
(``workset_record_reviewed`` counting the skip streak,
``workset_backoff_entry`` exposing it to the gate). Integration with
the rest of ``prepare_pr`` is covered structurally by the existing
prepare-pr tests; the gate is a single early-return.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import workset  # noqa: E402


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
        entry["last_llm_decision"] = "approve"
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


class WorksetBackoffRecordTests(unittest.TestCase):
    """``workset_record_reviewed`` is the single funnel after every LLM
    call. ``"skip"`` increments the persisted counter, anything else
    resets it; an error verdict leaves no reusable backoff entry.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.args = Namespace(
            workset_dir=Path(tmp.name), owner="o", repo="r",
            forge_type="gitea", gcli_account=None,
        )
        fairy.workset_record_queued(
            self.args, "pr", number=42, title="t", html_url="")

    def _record(self, classification: str) -> None:
        decision = fairy.Decision(
            42, "t", "a", "-", "skip", "r", LAST_ACT, classification, "")
        fairy.workset_record_reviewed(
            self.args, "pr", decision,
            expected_updated_at="U", expected_head_ref=HEAD)

    def _entry(self) -> dict[str, object]:
        return fairy.workset_backoff_entry(self.args, "pr", 42)

    def test_skip_initializes_counter_to_one(self) -> None:
        self._record("skip")
        entry = self._entry()
        self.assertEqual(entry["last_llm_decision"], "skip")
        self.assertEqual(entry["last_llm_head_sha"], HEAD)
        self.assertEqual(entry["last_llm_last_activity_iso"], LAST_ACT_ISO)
        self.assertEqual(entry["consecutive_skip_count"], 1)
        self.assertIsInstance(entry["last_llm_at"], str)

    def test_repeated_skip_increments_counter(self) -> None:
        self._record("skip")
        self._record("skip")
        self.assertEqual(self._entry()["consecutive_skip_count"], 2)

    def test_non_skip_resets_counter(self) -> None:
        self._record("skip")
        self._record("approve")
        entry = self._entry()
        self.assertEqual(entry["last_llm_decision"], "approve")
        self.assertEqual(entry["consecutive_skip_count"], 0)

    def test_error_never_suppresses(self) -> None:
        self._record("skip")
        self._record("error")
        self.assertEqual(self._entry(), {})
        item = workset.load_item(fairy.workset_path(self.args, "pr", 42))
        self.assertEqual(item.state, workset.WorkState.ERROR)

    def test_roundtrip_skip_then_compute_suppresses(self) -> None:
        self._record("skip")
        entry = self._entry()
        last_at = fairy.iso_to_dt(entry["last_llm_at"])
        assert last_at is not None
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, LAST_ACT, last_at + timedelta(hours=1),
        )
        self.assertIsNotNone(result)
        result = fairy.compute_llm_skip_backoff(
            entry, HEAD, LAST_ACT, last_at + timedelta(hours=25),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
