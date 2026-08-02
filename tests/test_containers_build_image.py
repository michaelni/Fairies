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

Tests for ``containers/build_image.py`` CLI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINERS_DIR = REPO_ROOT / "containers"
for p in (str(REPO_ROOT), str(CONTAINERS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_image  # noqa: E402


class BuildImageMainTests(unittest.TestCase):
    def test_invokes_build_image_if_needed_with_defaults(self) -> None:
        with mock.patch.object(build_image, "build_image_if_needed") as bif:
            rc = build_image.main(["--ssh", "fairy@h"])
        self.assertEqual(0, rc)
        bif.assert_called_once()
        _, kwargs = bif.call_args
        self.assertEqual(build_image.DEFAULT_TAG, kwargs["image_tag"])
        self.assertEqual("Containerfile", kwargs["dockerfile"].name)
        self.assertEqual("fairy@h", kwargs["host"].ssh_dest)

    def test_force_passed_through(self) -> None:
        with mock.patch.object(build_image, "build_image_if_needed") as bif:
            rc = build_image.main(["--ssh", "fairy@h", "--force", "--tag", "my:1"])
        self.assertEqual(0, rc)
        self.assertTrue(bif.call_args.kwargs["force"])
        self.assertEqual("my:1", bif.call_args.kwargs["image_tag"])

    def test_ssh_identity_builds_remote_host(self) -> None:
        with mock.patch.object(build_image, "build_image_if_needed") as bif:
            build_image.main(["--ssh", "fairy@h", "--identity", "/k/id"])
        host = bif.call_args.kwargs["host"]
        self.assertEqual("fairy@h", host.ssh_dest)
        self.assertEqual("/k/id", host.identity)


if __name__ == "__main__":
    unittest.main()
