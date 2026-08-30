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

SVG files must never be uploaded to a vector store.

Regression: an ``.svg`` file silently poisoned vector store
vs_6a272732f2608191ab291f0ad1285a51 -- ``files.list`` started returning a
permanent 500 past the page that held it. Bisecting the store
(``tools/openai_vector_store_bisect_poison.py``) named the single offending
entry as that SVG. SVG is not in OpenAI's supported vector-store file types
(https://developers.openai.com/api/docs/assistants/tools/file-search), but it
is XML/text so the backend sniffed it as text, indexed it, and corrupted the
index page instead of cleanly rejecting it. ``svg``/``svgz`` had been wrongly
listed as supported, so the file was uploaded as a raw ``.svg``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_vector_store  # noqa: E402


class SvgNotUploadedTests(unittest.TestCase):
    def test_svg_extensions_not_supported(self) -> None:
        for ext in ("svg", "svgz"):
            with self.subTest(ext=ext):
                self.assertNotIn(ext, openai_vector_store.SUPPORTED_VECTOR_STORE_EXTENSIONS)
                self.assertNotIn(ext, openai_vector_store.TEXT_WRAPPED_VECTOR_STORE_EXTENSIONS)

    def test_svg_path_is_skipped(self) -> None:
        # None == skip upload entirely.
        for relpath in ("doc/diagram.svg", "a/b/icon.SVG", "x.svgz"):
            with self.subTest(relpath=relpath):
                self.assertIsNone(
                    openai_vector_store.get_vector_store_upload_suffix(
                        relpath, openai_vector_store.SUPPORTED_VECTOR_STORE_EXTENSIONS
                    )
                )


class UndocumentedExtensionsWrappedTests(unittest.TestCase):
    """Regression: on 2026-08-30 ``vector_stores.file_batches.create``
    rejected a batch with 400 ``unsupported_file`` ("Files with extensions
    [.h] are not supported for retrieval") after months of raw ``.h``
    uploads being accepted. Only extensions on
    https://developers.openai.com/api/docs/guides/tools-file-search may be
    uploaded raw; other text sources must be wrapped as ``.txt``.
    """

    def test_undocumented_source_extensions_wrap_to_txt(self) -> None:
        for relpath in ("a/b.h", "x.pl", "y.pm"):
            with self.subTest(relpath=relpath):
                self.assertEqual(
                    openai_vector_store.get_vector_store_upload_suffix(
                        relpath, openai_vector_store.SUPPORTED_VECTOR_STORE_EXTENSIONS
                    ),
                    "txt",
                )


if __name__ == "__main__":
    unittest.main()
