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

A minimal stand-in for the ``anthropic`` SDK so its reviewer can be
driven without network access. ``install_anthropic`` replaces the SDK
unless it is already imported.
"""

from __future__ import annotations

import sys
import types


def install_anthropic() -> None:
    if "anthropic" in sys.modules:
        return
    fake = types.ModuleType("anthropic")

    class _E(Exception):
        pass

    class APIConnectionError(_E):
        pass

    fake.Anthropic = object
    fake.APIConnectionError = APIConnectionError
    fake.APITimeoutError = type("APITimeoutError", (APIConnectionError,), {})
    for name in ("RateLimitError", "InternalServerError", "OverloadedError"):
        setattr(fake, name, type(name, (_E,), {}))
    sys.modules["anthropic"] = fake

