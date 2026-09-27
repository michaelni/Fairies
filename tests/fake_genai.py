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

A minimal stand-in for the ``google-genai`` SDK so the Gemini reviewer can
be imported and driven where the package is not installed. ``install()``
leaves a real installation alone.
"""

from __future__ import annotations

import sys
import types

_TYPES = ("Candidate", "Content", "FunctionCall", "FunctionDeclaration",
          "FunctionResponse", "GenerateContentConfig", "GenerateContentResponse",
          "GenerateContentResponseUsageMetadata", "HttpOptions", "HttpRetryOptions",
          "Part", "ThinkingConfig", "Tool")


class _Obj:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)

    def __getattr__(self, name: str) -> None:
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def install() -> None:
    try:
        import google.genai  # noqa: F401
        return
    except ImportError:
        pass
    genai_types = types.ModuleType("google.genai.types")
    for name in _TYPES:
        setattr(genai_types, name, type(name, (_Obj,), {}))
    genai = types.ModuleType("google.genai")
    genai.types = genai_types
    genai.Client = object
    google_pkg = sys.modules.get("google") or types.ModuleType("google")
    google_pkg.genai = genai
    sys.modules.update({"google": google_pkg, "google.genai": genai,
                        "google.genai.types": genai_types})
