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

halt_marker: the operator's halt file round-trips."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import halt_marker  # noqa: E402


class HaltMarkerTests(unittest.TestCase):
    def test_reason_persists_until_the_file_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = halt_marker.path(Path(tmp))
            self.assertEqual(marker, Path(tmp) / "halted")
            self.assertIsNone(halt_marker.reason(marker))
            halt_marker.halt(marker, "carol posted 'STOP' in https://forge/pr/1")
            self.assertEqual(halt_marker.reason(marker),
                             "carol posted 'STOP' in https://forge/pr/1")
            marker.unlink()
            self.assertIsNone(halt_marker.reason(marker))


if __name__ == "__main__":
    unittest.main()
