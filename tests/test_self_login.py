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

Knowing which login fairy posts under is not optional.

``self_login`` is what tells fairy's own comments apart from
everyone else's: the CI heads-up dedupe, the "already reviewed this"
check and @-mention matching all key on it. When it comes back None
those three degrade to "nobody has said anything", and fairy posts a
duplicate on every run.

On Forgejo the login comes from ``/user``. A GitHub App installation
token acts as the app rather than a user, so ``/user`` answers 403
there (observed 2026-07-28 against the forgejo-fairy app on
michaelni/testrepo) and the login -- ``forgejo-fairy[bot]`` -- is not
reachable from that token at all. ``--self-login`` is how the operator
supplies it, and a failed lookup warns rather than passing None on in
silence.
"""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402


def _args(**kw) -> SimpleNamespace:
    return SimpleNamespace(forge_type="gitea", gcli_account="", verbose=0,
                           self_login=None, **kw)


class SelfLoginTests(unittest.TestCase):

    def test_the_forge_answer_is_used_when_there_is_one(self) -> None:
        with mock.patch.object(forge_gcli, "gcli_api",
                               return_value={"login": "Forgejo_Fairy"}):
            self.assertEqual(forge_gcli.self_login(_args()), "Forgejo_Fairy")

    def test_the_override_wins_without_asking_the_forge(self) -> None:
        args = _args()
        args.self_login = "forgejo-fairy[bot]"
        def unreachable(*a, **k):
            raise AssertionError("/user must not be called when told the login")
        with mock.patch.object(forge_gcli, "gcli_api", unreachable):
            self.assertEqual(forge_gcli.self_login(args), "forgejo-fairy[bot]")

    def test_a_denied_lookup_warns_instead_of_going_quiet(self) -> None:
        def denied(*a, **k):
            raise RuntimeError("403: Resource not accessible by integration")
        with mock.patch.object(forge_gcli, "gcli_api", denied):
            with self.assertLogs(forge_gcli.logger, logging.WARNING) as caught:
                self.assertIsNone(forge_gcli.self_login(_args()))
        self.assertIn("--self-login", "\n".join(caught.output))

    def test_a_reply_without_a_login_is_not_treated_as_one(self) -> None:
        with mock.patch.object(forge_gcli, "gcli_api", return_value={"id": 7}):
            self.assertIsNone(forge_gcli.self_login(_args()))


if __name__ == "__main__":
    unittest.main()
