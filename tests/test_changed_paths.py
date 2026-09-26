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

Tests for ``extract_changed_paths_from_patch``.

The fixture is FFmpeg commit 07754f3f8f, trimmed: a PR whose first
commit deletes a file has that file at no commit of the series, so
the source bundle must not ask for it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import patch_util  # noqa: E402


PATCH_DELETE_FILE = """\
From 07754f3f8fdac85f4a73f463b3b68f0214dcabda Mon Sep 17 00:00:00 2001
From: Michael Niedermayer <michael@niedermayer.cc>
Date: Mon, 6 Jan 2020 13:38:33 +0100
Subject: [PATCH] remove tests/ref/lavf/fits

This appears to be forgotten in ac4b5d86222006fa71ffe5922e1a34f1422507d8

Signed-off-by: Michael Niedermayer <michael@niedermayer.cc>
---
 tests/ref/lavf/fits | 3 ---
 1 file changed, 3 deletions(-)
 delete mode 100644 tests/ref/lavf/fits

diff --git a/tests/ref/lavf/fits b/tests/ref/lavf/fits
deleted file mode 100644
index 489542b32b..0000000000
--- a/tests/ref/lavf/fits
+++ /dev/null
@@ -1,3 +0,0 @@
-ed9fd697d0d782df6201f6a2db184552 *./tests/data/lavf/graylavf.fits
-5328000 ./tests/data/lavf/graylavf.fits
-./tests/data/lavf/graylavf.fits CRC=0xbacf446c
-- 
2.43.0

"""


class ChangedPathsTests(unittest.TestCase):
    def test_deleted_file_is_not_a_changed_path(self) -> None:
        self.assertEqual(
            patch_util.extract_changed_paths_from_patch(PATCH_DELETE_FILE), [])


if __name__ == "__main__":
    unittest.main()
