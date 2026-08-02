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

``is_container_unhealthy_error`` recognizes every observed shape.

The "container is gone" failure surfaces from the OpenAI API in more
than one shape. Regression: a main-pass ``responses.create`` failed with
a 404 ``NotFoundError`` / "Container has expired." (observed live on
#22687) that the detector missed -- it only matched 400
``BadRequestError`` / "Container is expired.". The miss re-raised, exit
code 1, and released the dead container back into the pool as healthy,
so the next attempts re-claimed it and the whole 3/3 retry cascade
failed.

Fixtures are the exact strings seen in the logs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import httpx
from openai import BadRequestError, NotFoundError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_container  # noqa: E402


def _error(cls: type, status: int, message: str):
    body = {"error": {"message": message, "type": "invalid_request_error",
                      "param": None, "code": None}}
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, request=request, json=body)
    return cls(f"Error code: {status} - {body}", response=response, body=body)


class IsContainerUnhealthyErrorTests(unittest.TestCase):
    def test_recognizes_all_observed_unhealthy_shapes(self) -> None:
        cases = [
            (BadRequestError, 400, "Container is expired."),
            (BadRequestError, 400, "Container is not running."),
            (NotFoundError, 404, "Container has expired."),
        ]
        for cls, status, message in cases:
            with self.subTest(message=message):
                self.assertTrue(
                    openai_container.is_container_unhealthy_error(
                        _error(cls, status, message)
                    )
                )

    def test_unrelated_bad_request_is_not_unhealthy(self) -> None:
        # A real 400 that is NOT about the container must not be treated
        # as a self-healable container failure.
        exc = _error(
            BadRequestError, 400,
            "code_interpreter and shell with an OpenAI-managed container "
            "cannot be used together at the same time.",
        )
        self.assertFalse(openai_container.is_container_unhealthy_error(exc))

    def test_non_openai_exception_is_not_unhealthy(self) -> None:
        self.assertFalse(
            openai_container.is_container_unhealthy_error(RuntimeError("boom"))
        )


if __name__ == "__main__":
    unittest.main()
