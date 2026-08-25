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

diff_render: git patch text to styled lines, on real ffmpeg samples."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diff_render  # noqa: E402
import tui_core  # noqa: E402


def line_text(line: tui_core.StyledLine) -> str:
    return "".join(t for _, t in line)


def styles(lines: list[tui_core.StyledLine]) -> set[str]:
    return {s for line in lines for s, _ in line}


# ffmpeg.git 2c18311d59ac by Michael Niedermayer, byte for byte as
# ``git format-patch --stdout`` emits it.
C_PATCH = """\
From 2c18311d59ac8b1a4ca2255750836b892179d27a Mon Sep 17 00:00:00 2001
From: Michael Niedermayer <michael@niedermayer.cc>
Date: Wed, 22 Jul 2026 22:46:55 +0200
Subject: [PATCH] avcodec/dovi_rpuenc: normalize vdr_dm_metadata_present to 0/1

---
 libavcodec/dovi_rpuenc.c | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/libavcodec/dovi_rpuenc.c b/libavcodec/dovi_rpuenc.c
index 8b7a74f313..d0abcc0d9e 100644
--- a/libavcodec/dovi_rpuenc.c
+++ b/libavcodec/dovi_rpuenc.c
@@ -726,7 +726,7 @@ int ff_dovi_rpu_generate(DOVIContext *s, const AVDOVIMetadata *metadata,
             return AVERROR(ENOMEM);
     }
\x20
-    vdr_dm_metadata_present = memcmp(color, &ff_dovi_color_default, sizeof(*color));
+    vdr_dm_metadata_present = !!memcmp(color, &ff_dovi_color_default, sizeof(*color));
     if (metadata->num_ext_blocks)
         vdr_dm_metadata_present = 1;
\x20
--\x20
2.43.0

"""

# One Makefile and one nasm file diff, from ffmpeg.git 7c944b0a3670
# and 26ea142658a8, both by Michael Niedermayer.
MULTI_FILE_DIFF = """\
diff --git a/tests/checkasm/x86/Makefile b/tests/checkasm/x86/Makefile
index 0254c61935..befe088dcf 100644
--- a/tests/checkasm/x86/Makefile
+++ b/tests/checkasm/x86/Makefile
@@ -3,4 +3,4 @@ CHECKASMOBJS-$(HAVE_YASM) += x86/checkasm.o
 tests/checkasm/x86/%.o: tests/checkasm/x86/%.asm
 \t$(DEPYASM) $(YASMFLAGS) -I $(<D)/ -M -o $@ $< > $(@:.o=.d)
 \t$(YASM) $(YASMFLAGS) -I $(<D)/ -o $@ $<
-\t-$(STRIP) $(STRIPFLAGS) $@
+\t-$(STRIP) $(ASMSTRIPFLAGS) $@
diff --git a/libavcodec/x86/lossless_videoencdsp.asm b/libavcodec/x86/lossless_videoencdsp.asm
index a9c7a0a73c..4d79eee36b 100644
--- a/libavcodec/x86/lossless_videoencdsp.asm
+++ b/libavcodec/x86/lossless_videoencdsp.asm
@@ -87,7 +87,7 @@ cglobal diff_bytes, 4,5,2, dst, src1, src2, w
         jz     .end_%1%2
 %if mmsize > 16
     ; fall back to narrower xmm
-    %define regsize mmsize / 2
+    %define regsize (mmsize / 2)
     DIFF_BYTES_LOOP_PREP .setup_loop_gpr_aa, .end_aa
 .loop2_%1%2:
     DIFF_BYTES_LOOP_CORE %1, %2, xm0, xm1
"""


QUOTED_HUNK_PATCH = """\
From 2c18311d59ac8b1a4ca2255750836b892179d27a Mon Sep 17 00:00:00 2001
From: Michael Niedermayer <michael@niedermayer.cc>
Date: Wed, 22 Jul 2026 22:46:55 +0200
Subject: [PATCH] doc: explain the rejected hunk

The submitted patch carried this hunk:

@@ -1,2 +1,2 @@ some_function
the line count in that header was wrong
which git am rejects

---
 f.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/f.txt b/f.txt
index 0000001..0000002 100644
--- a/f.txt
+++ b/f.txt
@@ -1 +1 @@
-old line
+new line
"""


class DiffViewTests(unittest.TestCase):
    def test_piecewise_slices_equal_the_full_render(self) -> None:
        full = diff_render.render_diff(C_PATCH)
        view = diff_render.DiffView(C_PATCH)
        self.assertEqual(len(view), len(full))
        self.assertEqual([line for i in range(0, len(view), 3)
                          for line in view[i:i + 3]], full)

    def sections_equal_the_lines_starting(self, patch: str, style: str,
                                          prefix: str) -> None:
        lines = patch.split("\n")
        self.assertEqual(diff_render.DiffView(patch).sections(style),
                         [i for i, line in enumerate(lines)
                          if line.startswith(prefix)])

    def test_sections_locate_the_commits_of_a_series(self) -> None:
        self.sections_equal_the_lines_starting(
            C_PATCH + QUOTED_HUNK_PATCH, "diff_commit", "From 2c18311d")

    def test_sections_locate_the_files_of_a_plain_diff(self) -> None:
        self.sections_equal_the_lines_starting(
            MULTI_FILE_DIFF, "diff_file", "diff --git ")

    def test_sections_locate_the_hunks(self) -> None:
        self.sections_equal_the_lines_starting(
            MULTI_FILE_DIFF, "diff_hunk", "@@ ")

    def test_a_hunk_header_quoted_in_a_commit_message_is_no_section(
            self) -> None:
        self.assertEqual(
            diff_render.DiffView(QUOTED_HUNK_PATCH).sections("diff_hunk"),
            [QUOTED_HUNK_PATCH.split("\n").index("@@ -1 +1 @@")])

    def test_sections_render_no_hunk(self) -> None:
        with mock.patch.object(diff_render, "_render_hunk") as render:
            view = diff_render.DiffView(MULTI_FILE_DIFF)
            self.assertEqual(len(view.sections("diff_hunk")), 2)
        render.assert_not_called()

    def test_only_hunks_under_a_slice_are_rendered(self) -> None:
        rendered = []
        real = diff_render._render_hunk

        def counting(body, lexer):
            rendered.append(len(body))
            return real(body, lexer)

        with mock.patch.object(diff_render, "_render_hunk",
                               side_effect=counting):
            view = diff_render.DiffView(MULTI_FILE_DIFF)
            self.assertGreater(len(view), 0)
            self.assertEqual(rendered, [])
            view[len(view) - 2:]
            self.assertEqual(len(rendered), 1)
            view[len(view) - 2:]
            self.assertEqual(len(rendered), 1)
            list(view)
            self.assertEqual(len(rendered), 2)


class RenderDiffTests(unittest.TestCase):
    def find(self, lines: list[tui_core.StyledLine],
             text: str) -> tui_core.StyledLine:
        for line in lines:
            if text in line_text(line):
                return line
        raise AssertionError(f"no rendered line contains {text!r}")

    def test_text_round_trips_unchanged(self) -> None:
        lines = diff_render.render_diff(C_PATCH)
        self.assertEqual([line_text(l) for l in lines],
                         C_PATCH.split("\n")[:len(lines)])

    def test_emitted_styles_stay_in_the_documented_set(self) -> None:
        for patch in (C_PATCH, MULTI_FILE_DIFF):
            self.assertLessEqual(styles(diff_render.render_diff(patch)),
                                 diff_render.DIFF_STYLES)

    def test_header_lines_get_their_styles(self) -> None:
        lines = diff_render.render_diff(C_PATCH)
        self.assertEqual(styles([self.find(lines, "From 2c18311d")]),
                         {"diff_commit"})
        self.assertEqual(styles([self.find(lines, "Subject:")]), {"bold"})
        self.assertEqual(styles([self.find(lines, "diff --git")]),
                         {"diff_file"})
        self.assertEqual(styles([self.find(lines, "@@ -726,7")]),
                         {"diff_hunk"})
        self.assertEqual(styles([self.find(lines, "index 8b7a74f313")]),
                         {"diff_meta"})

    def test_diffstat_counts_are_colored(self) -> None:
        stat = self.find(diff_render.render_diff(C_PATCH),
                         "dovi_rpuenc.c | 2 ")
        self.assertIn(("sc_good", "+"), stat)
        self.assertIn(("sc_bad", "-"), stat)

    def test_add_del_backgrounds_and_word_marks(self) -> None:
        lines = diff_render.render_diff(C_PATCH)
        minus = self.find(lines, "-    vdr_dm_metadata_present =")
        plus = self.find(lines, "+    vdr_dm_metadata_present =")
        self.assertTrue(all(s.startswith("df_del") for s, _ in minus))
        self.assertTrue(all(s.startswith("df_add") for s, _ in plus))
        self.assertEqual("".join(t for s, t in plus
                                 if s.startswith("df_addhl")), "!!")

    def test_commit_message_lines_are_plain_text(self) -> None:
        lines = diff_render.render_diff(C_PATCH)
        self.assertEqual(self.find(lines, "2.43.0"), [("text", "2.43.0")])

    def test_a_hunk_quoted_in_the_commit_message_stays_text(self) -> None:
        lines = diff_render.render_diff(QUOTED_HUNK_PATCH)
        self.assertEqual([line_text(l) for l in lines],
                         QUOTED_HUNK_PATCH.split("\n")[:len(lines)])
        self.assertEqual(self.find(lines, "@@ -1,2"),
                         [("text", "@@ -1,2 +1,2 @@ some_function")])
        self.assertEqual(styles([self.find(lines, "@@ -1 +1 @@")]),
                         {"diff_hunk"})

    def test_context_lines_get_syntax_foregrounds(self) -> None:
        lines = diff_render.render_diff(C_PATCH)
        ctx = self.find(lines, "    if (metadata->num_ext_blocks)")
        self.assertIn(("df_ctx_kw", "if"), ctx)
        self.assertIn(("df_ctx_kw", "return"),
                      self.find(lines, "return AVERROR(ENOMEM);"))

    def test_the_lexer_follows_the_file_headers(self) -> None:
        lines = diff_render.render_diff(MULTI_FILE_DIFF)
        make = self.find(lines, "tests/checkasm/x86/%.o:")
        self.assertIn("df_ctx_fn", styles([make]))
        # ``;`` opens a comment for nasm and for no Makefile.
        nasm = self.find(lines, "fall back to narrower xmm")
        self.assertIn("df_ctx_com", styles([nasm]))
        marked = self.find(lines, "-       -$(STRIP)")
        self.assertEqual("".join(t for s, t in marked
                                 if s.startswith("df_delhl")), "STRIPFLAGS")

    def test_a_bare_carriage_return_does_not_shift_hunk_styling(self) -> None:
        patch = ("diff --git a/f.c b/f.c\n--- a/f.c\n+++ b/f.c\n"
                 "@@ -1,3 +1,3 @@\n"
                 " int x;\rint y;\n"
                 " /* a comment */\n"
                 "-int b;\n"
                 "+int c;\n")
        lines = diff_render.render_diff(patch)
        self.assertIn(("df_ctx_com", "/* a comment */"),
                      self.find(lines, "a comment"))
        self.assertIn(("df_del_ty", "int"), self.find(lines, "int b;"))
        self.assertIn(("df_add_ty", "int"), self.find(lines, "int c;"))

    def test_multiline_constructs_keep_their_style_across_lines(self) -> None:
        patch = ("diff --git a/f.c b/f.c\n--- a/f.c\n+++ b/f.c\n"
                 "@@ -1,2 +1,4 @@\n"
                 " \n"
                 " int x;\n"
                 "+/* one\n"
                 "+   two */\n")
        lines = diff_render.render_diff(patch)
        # the blank first context line must not shift the row mapping
        self.assertIn(("df_ctx_ty", "int"), self.find(lines, "int x;"))
        self.assertEqual(styles([self.find(lines, "two */")]),
                         {"df_add_tx", "df_add_com"})

    def test_tabs_expand_consistently(self) -> None:
        lines = diff_render.render_diff("@@ -1 +1 @@\n-\tab\n+\tac\n")
        self.assertEqual(line_text(lines[1]), "-       ab")
        self.assertEqual(line_text(lines[2]), "+       ac")


if __name__ == "__main__":
    unittest.main()
