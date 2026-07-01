"""SVG files must never be uploaded to a vector store.

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


if __name__ == "__main__":
    unittest.main()
