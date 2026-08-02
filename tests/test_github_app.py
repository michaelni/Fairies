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

App auth is opt-in and must leave token deployments alone.

gcli takes its token from a config file and offers no flag or
environment variable for one, so authenticating as a GitHub App means
writing a minted token somewhere gcli will read. Two things matter and
are pinned here: a deployment that passes a static token must see no
change at all, and the operator's own gcli config must never be the
file that gets a one-hour token written into it.

The minting itself is covered by ``test_github_write_live``, which runs
against a real installation; these are the parts that must hold with no
network.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import github_app  # noqa: E402


def _args(**kw) -> SimpleNamespace:
    base = dict(gcli_account="", verbose=0, github_app_id=None,
                github_app_key=None, github_app_installation=None)
    base.update(kw)
    return SimpleNamespace(**base)


class OptInTests(unittest.TestCase):

    def setUp(self) -> None:
        github_app._state.clear()
        github_app._accounts.clear()
        self.addCleanup(github_app._state.clear)
        self.addCleanup(github_app._accounts.clear)

    def test_no_app_configured_means_no_environment_and_no_minting(self) -> None:  # noqa: E501
        def unreachable(*a, **k):
            raise AssertionError("must not reach the network")
        with mock.patch.object(github_app, "_api", unreachable):
            self.assertIsNone(github_app.gcli_env(_args()))

    def test_an_id_without_a_key_is_not_app_auth(self) -> None:
        self.assertIsNone(github_app.gcli_env(_args(github_app_id="1")))

    def test_a_key_without_an_id_is_not_app_auth(self) -> None:
        self.assertIsNone(
            github_app.gcli_env(_args(github_app_key=Path("/nonexistent.pem"))))


class MintedConfigTests(unittest.TestCase):

    def setUp(self) -> None:
        github_app._state.clear()
        github_app._accounts.clear()
        self.addCleanup(github_app._state.clear)
        self.addCleanup(github_app._accounts.clear)
        self.minted: list[str] = []

        def fake_mint(app_id, key_path, installation):
            self.minted.append(installation)
            return "ghs_fake", datetime.now(timezone.utc) + timedelta(hours=1)

        self.mint = mock.patch.object(github_app, "_mint", fake_mint)
        self.mint.start()
        self.addCleanup(self.mint.stop)
        self.inst = mock.patch.object(
            github_app, "_installation_id", lambda *a: "149374420")
        self.inst.start()
        self.addCleanup(self.inst.stop)

    def _env(self, **kw):
        return github_app.gcli_env(_args(
            github_app_id="4406646", github_app_key=Path("/k.pem"), **kw))

    def test_the_token_goes_to_a_private_config_not_the_operators(self) -> None:
        env = self._env()
        home = Path(env["XDG_CONFIG_HOME"])
        self.assertNotEqual(home, Path.home() / ".config")
        config = home / "gcli" / "config"
        self.assertIn("ghs_fake", config.read_text())
        self.assertEqual(config.stat().st_mode & 0o077, 0,
                         "a token file must not be group or world readable")

    def test_the_account_name_matches_what_gcli_is_told_to_use(self) -> None:
        env = self._env(gcli_account="fairy-gh")
        config = (Path(env["XDG_CONFIG_HOME"]) / "gcli" / "config").read_text()
        self.assertTrue(config.startswith("fairy-gh {"), config)

    def test_a_live_token_is_not_reminted_on_every_call(self) -> None:
        self._env()
        self._env()
        self._env()
        self.assertEqual(len(self.minted), 1)

    def test_a_token_near_expiry_is_replaced(self) -> None:
        self._env()
        key, (token, _) = next(iter(github_app._accounts.items()))
        github_app._accounts[key] = (token, datetime.now(timezone.utc)
            + timedelta(seconds=github_app.REFRESH_MARGIN_SECONDS - 1))
        self._env()
        self.assertEqual(len(self.minted), 2)

    def test_an_unnamed_account_becomes_gclis_default(self) -> None:
        # gcli_prefix sends no -a when no account was named, so without a
        # defaults entry gcli would find no token and go out anonymous.
        config = (Path(self._env()["XDG_CONFIG_HOME"])
                  / "gcli" / "config").read_text()
        self.assertIn("github-default-account=app", config)

    def test_a_named_account_does_not_claim_the_default(self) -> None:
        config = (Path(self._env(gcli_account="fairy-gh")["XDG_CONFIG_HOME"])
                  / "gcli" / "config").read_text()
        self.assertNotIn("github-default-account", config)

    def test_concurrent_callers_mint_once_and_never_see_a_partial_config(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            envs = list(pool.map(lambda _: self._env(), range(8)))
        self.assertEqual(len(self.minted), 1)
        for env in envs:
            config = (Path(env["XDG_CONFIG_HOME"]) / "gcli" / "config").read_text()
            self.assertIn("ghs_fake", config)

    def test_the_rest_of_the_environment_is_carried_through(self) -> None:
        with mock.patch.dict(os.environ, {"FAIRY_MARKER": "kept"}):
            self.assertEqual(self._env()["FAIRY_MARKER"], "kept")


class OperatorFlagsTests(unittest.TestCase):

    def test_the_flags_parse_into_the_names_gcli_env_reads(self) -> None:
        parser = argparse.ArgumentParser()
        github_app.add_github_app_args(parser)
        ns = parser.parse_args(
            ["--github-app-id", "4406646", "--github-app-key", "/k.pem",
             "--github-app-installation", "149374420"])
        self.assertEqual(ns.github_app_id, "4406646")
        self.assertEqual(ns.github_app_key, Path("/k.pem"))
        self.assertEqual(ns.github_app_installation, "149374420")


if __name__ == "__main__":
    unittest.main()
