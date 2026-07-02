"""Tests for submodule (gitlink) detection in patch parsing.

A submodule shows up in a unified diff in three distinguishable ways
depending on the change kind:

- new submodule:  ``new file mode 160000``
- deleted submodule:  ``deleted file mode 160000``
- bumped commit:  ``index <old>..<new> 160000`` plus ``-Subproject
  commit <old>`` / ``+Subproject commit <new>`` hunk lines

The wrapper has to recognise all three and exclude the gitlink path
from the source bundle, because ``load_source_bundle_texts`` cannot
``git show`` a submodule and would otherwise crash with
``RuntimeError: missing source for <submodule>`` (as observed for
FFmpeg's ``tests/checkasm/ext`` when the path was first added in
PR #22546). The earlier detector only fired on the new/deleted-mode
header and on a literal ``\\nSubproject commit `` substring, so a
*bumped* submodule (whose body starts with ``-Subproject commit ...``,
i.e. a sign-prefixed line) slipped through silently. This file pins
all three cases.

The patches below are deliberately formatted the way ``git
format-patch`` emits them, including the ``-- \\n<version>\\nFrom
<sha>`` boundary between consecutive commits, so the regression test
also covers ``_diff_block_own_body``'s job of truncating each split
diff block at the next-commit boundary.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_pr_review_wrapper as wrapper  # noqa: E402
import patch_util  # noqa: E402


PATCH_ADD_SUBMODULE = """\
From abcdef1234567890abcdef1234567890abcdef12 Mon Sep 17 00:00:00 2001
From: Alice <alice@example.com>
Date: Sun, 26 Apr 2026 12:00:00 +0000
Subject: [PATCH] add ext submodule and a regular file

---
 .gitmodules         | 3 +++
 tests/regular.c     | 1 +
 tests/checkasm/ext  | 1 +
 3 files changed, 5 insertions(+)
 create mode 100644 .gitmodules
 create mode 100644 tests/regular.c
 create mode 160000 tests/checkasm/ext

diff --git a/.gitmodules b/.gitmodules
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/.gitmodules
@@ -0,0 +1,3 @@
+[submodule "tests/checkasm/ext"]
+\tpath = tests/checkasm/ext
+\turl = https://example.org/ext.git
diff --git a/tests/regular.c b/tests/regular.c
new file mode 100644
index 0000000..2222222
--- /dev/null
+++ b/tests/regular.c
@@ -0,0 +1 @@
+int x = 1;
diff --git a/tests/checkasm/ext b/tests/checkasm/ext
new file mode 160000
index 0000000..deadbee
--- /dev/null
+++ b/tests/checkasm/ext
@@ -0,0 +1 @@
+Subproject commit deadbeefcafebabe1234567890abcdef00112233
--\x20
2.40.0

"""


PATCH_UPDATE_SUBMODULE = """\
From cafebabecafebabecafebabecafebabecafebabe Mon Sep 17 00:00:00 2001
From: Bob <bob@example.com>
Date: Sun, 26 Apr 2026 13:00:00 +0000
Subject: [PATCH] bump ext

---
 tests/checkasm/ext | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/tests/checkasm/ext b/tests/checkasm/ext
index aaaaaaa..bbbbbbb 160000
--- a/tests/checkasm/ext
+++ b/tests/checkasm/ext
@@ -1 +1 @@
-Subproject commit aaaaaaa1111111122222222333333334444444455
+Subproject commit bbbbbbb1111111122222222333333334444444455
--\x20
2.40.0

"""


PATCH_REMOVE_SUBMODULE = """\
From 1234567812345678123456781234567812345678 Mon Sep 17 00:00:00 2001
From: Carol <carol@example.com>
Date: Sun, 26 Apr 2026 14:00:00 +0000
Subject: [PATCH] drop ext

---
 .gitmodules        | 3 ---
 tests/checkasm/ext | 1 -
 2 files changed, 4 deletions(-)
 delete mode 160000 tests/checkasm/ext

diff --git a/.gitmodules b/.gitmodules
deleted file mode 100644
index 1111111..0000000
--- a/.gitmodules
+++ /dev/null
@@ -1,3 +0,0 @@
-[submodule "tests/checkasm/ext"]
-\tpath = tests/checkasm/ext
-\turl = https://example.org/ext.git
diff --git a/tests/checkasm/ext b/tests/checkasm/ext
deleted file mode 160000
index aaaaaaa..0000000
--- a/tests/checkasm/ext
+++ /dev/null
@@ -1 +0,0 @@
-Subproject commit aaaaaaa1111111122222222333333334444444455
--\x20
2.40.0

"""


PATCH_TWO_COMMITS = """\
From aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa Mon Sep 17 00:00:00 2001
From: Alice <alice@example.com>
Date: Sun, 26 Apr 2026 10:00:00 +0000
Subject: [PATCH 1/2] add regular file

---
 tests/fate.sh | 1 +
 1 file changed, 1 insertion(+)

diff --git a/tests/fate.sh b/tests/fate.sh
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/tests/fate.sh
@@ -0,0 +1 @@
+#!/bin/sh
--\x20
2.40.0

From bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb Mon Sep 17 00:00:00 2001
From: Alice <alice@example.com>
Date: Sun, 26 Apr 2026 10:30:00 +0000
Subject: [PATCH 2/2] add ext submodule

---
 tests/checkasm/ext | 1 +
 1 file changed, 1 insertion(+)
 create mode 160000 tests/checkasm/ext

diff --git a/tests/checkasm/ext b/tests/checkasm/ext
new file mode 160000
index 0000000..deadbee
--- /dev/null
+++ b/tests/checkasm/ext
@@ -0,0 +1 @@
+Subproject commit deadbeefcafebabe1234567890abcdef00112233
--\x20
2.40.0

"""


class SubmoduleDetectionTests(unittest.TestCase):
    def test_added_submodule_path_is_in_set(self) -> None:
        self.assertEqual(
            patch_util.extract_submodule_paths_from_patch(PATCH_ADD_SUBMODULE),
            {"tests/checkasm/ext"},
        )

    def test_updated_submodule_path_is_in_set(self) -> None:
        """Regression: bumped submodules emit ``index <old>..<new> 160000``
        plus sign-prefixed ``Subproject commit`` lines, neither of which
        the older substring-only detector matched."""
        self.assertEqual(
            patch_util.extract_submodule_paths_from_patch(PATCH_UPDATE_SUBMODULE),
            {"tests/checkasm/ext"},
        )

    def test_removed_submodule_path_is_in_set(self) -> None:
        self.assertEqual(
            patch_util.extract_submodule_paths_from_patch(PATCH_REMOVE_SUBMODULE),
            {"tests/checkasm/ext"},
        )

    def test_changed_paths_excludes_added_submodule(self) -> None:
        paths = patch_util.extract_changed_paths_from_patch(PATCH_ADD_SUBMODULE)
        self.assertNotIn("tests/checkasm/ext", paths)
        self.assertIn(".gitmodules", paths)
        self.assertIn("tests/regular.c", paths)

    def test_changed_paths_excludes_updated_submodule(self) -> None:
        paths = patch_util.extract_changed_paths_from_patch(PATCH_UPDATE_SUBMODULE)
        self.assertNotIn("tests/checkasm/ext", paths)


class SubmoduleChangesExtractionTests(unittest.TestCase):
    """Tests for the structured ``extract_submodule_changes_from_patch``
    helper that feeds the LLM-prompt metadata field."""

    def test_added_submodule_is_classified_with_new_commit_only(self) -> None:
        self.assertEqual(
            patch_util.extract_submodule_changes_from_patch(PATCH_ADD_SUBMODULE),
            [
                {
                    "path": "tests/checkasm/ext",
                    "action": "added",
                    "old_commit": None,
                    "new_commit": "deadbeefcafebabe1234567890abcdef00112233",
                }
            ],
        )

    def test_updated_submodule_carries_both_old_and_new_commit(self) -> None:
        self.assertEqual(
            patch_util.extract_submodule_changes_from_patch(PATCH_UPDATE_SUBMODULE),
            [
                {
                    "path": "tests/checkasm/ext",
                    "action": "updated",
                    "old_commit": "aaaaaaa1111111122222222333333334444444455",
                    "new_commit": "bbbbbbb1111111122222222333333334444444455",
                }
            ],
        )

    def test_removed_submodule_is_classified_with_old_commit_only(self) -> None:
        self.assertEqual(
            patch_util.extract_submodule_changes_from_patch(PATCH_REMOVE_SUBMODULE),
            [
                {
                    "path": "tests/checkasm/ext",
                    "action": "removed",
                    "old_commit": "aaaaaaa1111111122222222333333334444444455",
                    "new_commit": None,
                }
            ],
        )

    def test_format_patch_boundary_does_not_mislabel_regular_file(self) -> None:
        """Commit 2's ``create mode 160000`` summary line must not be
        attributed to commit 1's last regular file when the patch is
        split on ``^diff --git`` and parsed for submodule action."""
        self.assertEqual(
            patch_util.extract_submodule_changes_from_patch(PATCH_TWO_COMMITS),
            [
                {
                    "path": "tests/checkasm/ext",
                    "action": "added",
                    "old_commit": None,
                    "new_commit": "deadbeefcafebabe1234567890abcdef00112233",
                }
            ],
        )


def _make_request(*, patch: str, number: int) -> dict:
    return {
        "pull_request": {
            "number": number,
            "title": "test",
            "author": "alice",
            "html_url": f"https://example.org/pr/{number}",
            "base_ref": "master",
            "head_ref": "alice/test",
            "head_sha": "f" * 40,
            "additions": 1,
            "deletions": 0,
            "changed_files": 1,
            "auto_merge": None,
            "body": "",
        },
        "patch": patch,
        "patch_truncated": False,
        "discussion": [],
        "reviewer_username": "Forgejo_Fairy",
    }


class MakeUserTextSubmoduleMetadataTests(unittest.TestCase):
    """The main reviewer's prompt must surface submodule changes
    explicitly so the LLM does not have to spot a one-line ``Subproject
    commit ...`` hunk hidden inside a multi-thousand-line diff."""

    def test_make_user_text_surfaces_submodule_changes(self) -> None:
        text = wrapper.make_user_text(
            _make_request(patch=PATCH_ADD_SUBMODULE, number=12345),
            source_notes=[],
            source_files=[".gitmodules", "tests/regular.c"],
            patch_was_truncated=False,
        )
        self.assertIn('"submodule_changes":', text)
        self.assertIn('"path": "tests/checkasm/ext"', text)
        self.assertIn('"action": "added"', text)
        self.assertIn('"new_commit": "deadbeefcafebabe1234567890abcdef00112233"', text)

    def test_make_user_text_omits_submodule_changes_for_normal_pr(self) -> None:
        plain_patch = (
            "diff --git a/x.c b/x.c\n"
            "index 1111111..2222222 100644\n"
            "--- a/x.c\n"
            "+++ b/x.c\n"
            "@@ -1 +1,2 @@\n"
            " int x = 1;\n"
            "+int y = 2;\n"
        )
        text = wrapper.make_user_text(
            _make_request(patch=plain_patch, number=12346),
            source_notes=[],
            source_files=["x.c"],
            patch_was_truncated=False,
        )
        self.assertNotIn("submodule_changes", text)


class MakeTriageUserTextSubmoduleMetadataTests(unittest.TestCase):
    """The triage gatekeeper sees the same submodule callout, since
    deciding whether to engage the main reviewer is exactly when the
    "external code is being pulled in" signal matters most."""

    def test_make_triage_user_text_surfaces_submodule_changes(self) -> None:
        text = wrapper.make_triage_user_text(
            _make_request(patch=PATCH_UPDATE_SUBMODULE, number=22000),
            patch_was_truncated=False,
        )
        self.assertIn('"submodule_changes":', text)
        self.assertIn('"action": "updated"', text)
        self.assertIn('"old_commit": "aaaaaaa1111111122222222333333334444444455"', text)
        self.assertIn('"new_commit": "bbbbbbb1111111122222222333333334444444455"', text)


if __name__ == "__main__":
    unittest.main()
