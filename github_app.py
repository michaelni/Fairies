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

GitHub App authentication for gcli.

What belongs here: turning an App id plus its private key into the
short-lived installation token gcli needs, and keeping a gcli config
that carries it. What does NOT belong: anything about what the token is
then used for -- forge_gcli owns that.

gcli reads its token from a config file and has no flag or environment
variable for one, so an App token has to be written to a config gcli
will read. That config lives in a private directory this module owns,
handed to the gcli subprocess as XDG_CONFIG_HOME. It is never the
operator's own config: a token that expires in an hour has no business
being written there.

Installation tokens last an hour, so ``gcli_env`` re-mints when the
current one is close to expiry. Callers ask for the environment on
every gcli invocation and get a cached answer until then.

The JWT is signed by the openssl binary rather than a Python library:
signing an RS256 assertion is one subprocess call, and it keeps a
crypto dependency out of a tree that otherwise has none.

Public API (__all__):
    ``gcli_env`` -- environment for a gcli subprocess, or None when App
    auth is not configured
    ``add_github_app_args`` -- the three operator flags
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import logging
import os
import shutil
import subprocess
import threading
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["add_github_app_args", "gcli_env"]

logger = logging.getLogger("github_app")

API = "https://api.github.com"
# Re-mint this long before expiry so a request cannot set off with a
# token that dies mid-flight.
REFRESH_MARGIN_SECONDS = 300
JWT_LIFETIME_SECONDS = 540

# Reviews run on a thread pool (worker --parallel), so minting and the
# config rewrite are serialised and the file is swapped atomically: a
# gcli subprocess must never read a half-written token.
_lock = threading.Lock()
_state: dict[str, object] = {}
_accounts: dict[tuple[str, str], tuple[str, datetime]] = {}


def add_github_app_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--github-app-id",
        help="GitHub App id to authenticate as, instead of a static token.",
    )
    parser.add_argument(
        "--github-app-key",
        type=Path,
        help="PEM private key for --github-app-id.",
    )
    parser.add_argument(
        "--github-app-installation",
        help="Installation id to mint tokens for; discovered when the app "
             "has exactly one installation.",
    )


def _b64(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def _jwt(app_id: str, key_path: Path) -> str:
    now = int(time.time())
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64(json.dumps(
        {"iat": now - 60, "exp": now + JWT_LIFETIME_SECONDS, "iss": app_id},
    ).encode())
    signing_input = header + b"." + claims
    logger.debug("+ openssl dgst -sha256 -sign %s  # app_id=%s", key_path, app_id)
    signed = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(key_path)],
        input=signing_input, capture_output=True, check=True,
    ).stdout
    return (signing_input + b"." + _b64(signed)).decode()


def _api(path: str, token: str, method: str = "GET") -> object:
    request = urllib.request.Request(API + path, method=method)
    request.add_header("Authorization", "Bearer " + token)
    request.add_header("Accept", "application/vnd.github+json")
    logger.debug("+ %s %s", method, API + path)
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(
            f"github app: {method} {path} -> HTTP {exc.code}: {detail}"
        ) from exc


def _installation_id(app_id: str, key_path: Path, configured: str | None) -> str:
    if configured:
        return configured
    installations = _api("/app/installations", _jwt(app_id, key_path))
    if not isinstance(installations, list) or len(installations) != 1:
        found = [i.get("id") for i in installations] if isinstance(installations, list) else installations
        raise RuntimeError(
            f"github app {app_id}: expected exactly one installation to pick "
            f"automatically, found {found}. Pass --github-app-installation."
        )
    return str(installations[0]["id"])


def _mint(app_id: str, key_path: Path, installation: str) -> tuple[str, datetime]:
    minted = _api(f"/app/installations/{installation}/access_tokens",
                  _jwt(app_id, key_path), method="POST")
    expires = datetime.strptime(
        minted["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    logger.info("github app %s: minted installation token for %s, expires %s",
                app_id, installation, minted["expires_at"])
    return minted["token"], expires


def _config_dir() -> Path:
    existing = _state.get("dir")
    if existing is not None:
        return existing
    created = Path(tempfile.mkdtemp(prefix="fairy-gcli-app-"))
    created.chmod(0o700)
    atexit.register(shutil.rmtree, created, True)
    _state["dir"] = created
    return created


def _write_config() -> None:
    """Rewrite the private gcli config from ``_accounts``. Caller holds the lock.

    ``gcli_prefix`` only passes ``-a`` when an account was named, so an
    operator who gives only the app flags leaves gcli to its own default
    resolution; without a ``defaults`` entry it would find no token and
    fall back to unauthenticated requests -- reads throttled to the
    anonymous budget and every write refused. ``github-default-account``
    is what makes the tokenless invocation resolve.
    """
    config = _config_dir() / "gcli"
    config.mkdir(parents=True, exist_ok=True)
    config.chmod(0o700)
    body = ""
    default = _state.get("default_account")
    if default:
        body += f"defaults {{\n    github-default-account={default}\n}}\n\n"
    for (_, account), (token, _expires) in sorted(_accounts.items()):
        body += f"{account} {{\n    forge-type=github\n    token = {token}\n}}\n"
    scratch = config / "config.new"
    scratch.write_text(body, encoding="utf-8")
    scratch.chmod(0o600)
    scratch.replace(config / "config")


def gcli_env(args: argparse.Namespace) -> dict[str, str] | None:
    """Environment for a gcli subprocess, or None for static-token auth.

    Returns None unless both ``--github-app-id`` and ``--github-app-key``
    are set, so a deployment using a plain token is untouched. The
    account written here is named after ``args.gcli_account`` when there
    is one, so the ``gcli -a`` the rest of the code already passes
    resolves to the app; when there is none it also becomes gcli's
    default account, because no ``-a`` will be sent.
    """
    app_id = getattr(args, "github_app_id", None)
    key_path = getattr(args, "github_app_key", None)
    if not app_id or not key_path:
        return None

    named = getattr(args, "gcli_account", None)
    account = named or "app"
    with _lock:
        key = (str(app_id), account)
        held = _accounts.get(key)
        if (held is None
                or (held[1] - datetime.now(timezone.utc)).total_seconds()
                < REFRESH_MARGIN_SECONDS):
            installation = _installation_id(
                str(app_id), Path(key_path),
                getattr(args, "github_app_installation", None))
            _accounts[key] = _mint(str(app_id), Path(key_path), installation)
            if not named:
                _state["default_account"] = account
            _write_config()
        return {**os.environ, "XDG_CONFIG_HOME": str(_config_dir())}
