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

Every tracked ``*.py`` and ``*.sh`` file opens with the byte-exact license
header (after an optional shebang line): one changed, missing or added byte
inside the header is a failure.  Other file types (fixtures, JSON, markdown)
cannot carry the header and are out of scope.
"""
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_TEXT = """\
Copyright (C) 2026 Michael Niedermayer

This file is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

This file is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License version 2 for more details.

Additional permission:

Michael Niedermayer is permitted to relicense this file, in whole or
in part, under any version of the GNU General Public License, the GNU
Affero General Public License, or the GNU Lesser General Public License
published by the Free Software Foundation.

This additional permission is personal to Michael Niedermayer.  It is
not transferable and does not grant any other person permission to
relicense this file under a different license.

This additional permission may be removed from modified copies of this
file.  Removal of this additional permission does not affect the
licensing of the file under the GNU General Public License version 2.
"""

_LINES = _TEXT.splitlines()
_PY_HEADER = ('"""\n/*\n'
              + "".join(f" * {l}".rstrip() + "\n" for l in _LINES)
              + ' */\n').encode()
_SH_HEADER = "".join(f"# {l}".rstrip() + "\n" for l in _LINES).encode()


def _deviates(data: bytes, header: bytes) -> bool:
    if data.startswith(b"#!"):
        data = data.partition(b"\n")[2]
    return not data.startswith(header)


@unittest.skipUnless(shutil.which("git"), "git required")
class LicenseHeaderTests(unittest.TestCase):
    def test_every_tracked_file_opens_with_the_exact_header(self) -> None:
        offenders = []
        for pattern, header in (("*.py", _PY_HEADER), ("*.sh", _SH_HEADER)):
            files = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "ls-files", pattern],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            self.assertTrue(files, f"git ls-files {pattern} returned nothing")
            offenders += [f for f in files
                          if _deviates((REPO_ROOT / f).read_bytes(), header)]
        self.assertEqual([], offenders,
                         "license header missing, misplaced, or not byte-identical "
                         "to the canonical text")

    def test_any_single_byte_deviation_is_an_offense(self) -> None:
        for header in (_PY_HEADER, _SH_HEADER):
            for i in range(len(header)):
                flipped = header[:i] + bytes([header[i] ^ 1]) + header[i + 1:]
                self.assertTrue(_deviates(flipped, header))
                self.assertTrue(_deviates(header[:i] + header[i + 1:], header))


if __name__ == "__main__":
    unittest.main()
