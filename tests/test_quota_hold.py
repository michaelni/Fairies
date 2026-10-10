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
llm_review_api's quota hold: ensure_quota passes an account retry_backoff
knows nothing about, quota_exhausted records the exhaustion and the next
ensure_quota of any process raises QuotaExhausted due when the record
says, a reset the server named stretches that, passes meeting the same
exhaustion count it once, and a repeat after the hold doubles it.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_review_api import (QUOTA_BACKOFF_BASE_S, QuotaExhausted,  # noqa: E402
                            ensure_quota, quota_exhausted)
from test_retry_backoff import isolate  # noqa: E402


class QuotaHoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = isolate(self)

    def test_an_exhaustion_holds_the_account_until_due(self) -> None:
        ensure_quota("codex/home")
        exc = quota_exhausted("codex/home", "plan usage limit reached")
        self.assertEqual("plan usage limit reached", str(exc))
        self.assertAlmostEqual(time.time() + QUOTA_BACKOFF_BASE_S, exc.due_at, delta=2)
        with self.assertRaises(QuotaExhausted) as held:
            ensure_quota("codex/home")
        self.assertAlmostEqual(exc.due_at, held.exception.due_at, delta=2)
        self.assertRegex(str(held.exception),
                         "^BACKOFF, no provider call: .*plan usage limit reached")
        self.assertEqual(1, len(list(self.records.iterdir())))
        ensure_quota("zai/ZAI_API_KEY")

    def test_the_servers_reset_outlasts_the_base_wait(self) -> None:
        reset = time.time() + 5 * 3600
        self.assertAlmostEqual(reset, quota_exhausted("zai/ZAI_API_KEY", "[1308]", reset).due_at,
                               delta=2)
        with self.assertRaises(QuotaExhausted) as held:
            ensure_quota("zai/ZAI_API_KEY")
        self.assertAlmostEqual(reset, held.exception.due_at, delta=2)

    def test_passes_in_flight_together_count_one_exhaustion(self) -> None:
        for _ in range(4):
            quota_exhausted("grok/home", "402 Payment Required")
        record, = self.records.iterdir()
        self.assertEqual(1, len(record.read_text(encoding="utf-8").splitlines()))
        self.assertAlmostEqual(time.time() + QUOTA_BACKOFF_BASE_S,
                               quota_exhausted("grok/home", "402").due_at, delta=2)

    def test_a_repeat_after_the_hold_doubles_it(self) -> None:
        quota_exhausted("grok/home", "402 Payment Required")
        record, = self.records.iterdir()
        os.utime(record, times=(time.time() - QUOTA_BACKOFF_BASE_S - 1,) * 2)
        ensure_quota("grok/home")
        exc = quota_exhausted("grok/home", "402 Payment Required")
        self.assertEqual(2, len(record.read_text(encoding="utf-8").splitlines()))
        self.assertAlmostEqual(time.time() + 2 * QUOTA_BACKOFF_BASE_S, exc.due_at, delta=2)
        with self.assertRaises(QuotaExhausted) as held:
            ensure_quota("grok/home")
        self.assertAlmostEqual(exc.due_at, held.exception.due_at, delta=2)


if __name__ == "__main__":
    unittest.main()
