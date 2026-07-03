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

The one reviewer interface the OpenAI / Anthropic / GLM reviewers and the
review pipeline all speak.

What belongs here: the provider-neutral review vocabulary (``Review``,
``ReviewContext``, the ``Reviewer`` base class), the classification
constants, the review-output JSON schema with its validator
(``REVIEW_SCHEMA``, ``check_schema``, ``validate_review``), the shared
``BadModelOutput`` error, and ``run_parallel`` (the single fan-out
helper).

What does NOT belong: any provider/SDK-specific code (that lives in the
``*_reviewer`` modules), prompt text (``llm_prompt``), or pipeline wiring
(``review_pipeline``). This is a leaf module: it must not import any of
those.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from common import JsonObject
from podman_host import ContainerShellSession

__all__ = [
    "CLASSIFICATIONS",
    "ISSUE_CLASSIFICATIONS",
    "TERMINAL_ROUTES",
    "ENGAGE",
    "REVIEW_SCHEMA",
    "Z_AI_ANTHROPIC_URL",
    "BadModelOutput",
    "Review",
    "ReviewContext",
    "Reviewer",
    "SchemaError",
    "check_schema",
    "run_parallel",
    "validate_review",
]

logger = logging.getLogger(__name__)


# The full classification vocabulary a reviewer may emit. Single source of
# truth: the OpenAI/Anthropic output schemas and ``Review`` all read it.
CLASSIFICATIONS = (
    "ok_approve",
    "minor_issues_approve",
    "moderate_issues_comment",
    "major_request_changes",
    "helpful_reply",
    "skip",
)

# Classifications that mean "this reviewer found something worth raising".
ISSUE_CLASSIFICATIONS = ("moderate_issues_comment", "major_request_changes")

# Verdicts that are already final: when a stage returns one of these the
# pipeline stops and posts it as-is (no further models, no combine).
TERMINAL_ROUTES = ("skip", "helpful_reply")

# Non-emittable "keep going" marker a triager returns when it routes
# ``engage``. Never a valid posted classification and never reaches
# stdout; the pipeline treats it purely as "continue to the next stage".
ENGAGE = "engage"

# z.ai's Anthropic-compatible Messages endpoint. GLM is reached by pointing
# the Anthropic reviewer at this base URL with a z.ai key.
Z_AI_ANTHROPIC_URL = "https://api.z.ai/api/anthropic"


REVIEW_SCHEMA = {
    "name": "pr_review_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "classification": {
                "type": "string",
                "description": (
                    "Overall disposition of the pull request."
                ),
                "enum": list(CLASSIFICATIONS),
            },
            "message": {
                "type": "string",
                "description": (
                    "detailed Markdown comment body to post to Forgejo. "
                    "Must be empty for ok_approve. "
                    "Do not include HTML or markdown fences."
                ),
            },
        },
        "required": ["classification", "message"],
    },
}


class SchemaError(ValueError):
    """The model's JSON did not match the ``json_schema`` we requested.

    OpenAI ``strict`` structured output is best-effort, not a guarantee:
    it can silently slip (observed: emitting ``class`` instead of
    ``classification``). So every response is re-validated against the
    exact schema we sent before we trust it.
    """


class BadModelOutput(Exception):
    """The model's JSON did not parse / match ``REVIEW_SCHEMA``.

    Raised by the reviewers so the entrypoint exits
    ``EXIT_BAD_MODEL_OUTPUT`` and the caller retries the run.
    """


_JSON_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "object": dict,
    "array": list,
    "null": type(None),
}


def _matches_json_type(value: object, json_type: str) -> bool:
    # ``bool`` is a subclass of ``int``; keep the two distinct so a
    # boolean never satisfies integer/number and vice versa.
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type in ("integer", "number"):
        return isinstance(value, _JSON_PY_TYPES[json_type]) and not isinstance(value, bool)
    return isinstance(value, _JSON_PY_TYPES[json_type])


def check_schema(value: object, schema: dict[str, object], path: str = "$") -> None:
    """Validate ``value`` against the JSON Schema subset our request
    schemas use: ``type`` (incl. nullable unions), ``enum``, object
    ``properties`` / ``required`` / ``additionalProperties: false``, and
    array ``items`` / ``maxItems``. Raise :class:`SchemaError` at the first mismatch,
    naming the offending location. Deliberately not a general validator
    -- it covers exactly what ``REVIEW_SCHEMA`` / ``build_triage_schema``
    emit, so the next reader can trust the two stay in lockstep.
    """
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} not in {schema['enum']}")
    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared] if declared else []
    if types and not any(_matches_json_type(value, t) for t in types):
        raise SchemaError(f"{path}: expected type {declared!r}, got {type(value).__name__}")
    if "object" in types and isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise SchemaError(f"{path}: unexpected keys {unexpected}")
        for key, subschema in properties.items():
            if key in value:
                check_schema(value[key], subschema, f"{path}.{key}")
    if "array" in types and isinstance(value, list):
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise SchemaError(f"{path}: {len(value)} items exceed maxItems {max_items}")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                check_schema(item, item_schema, f"{path}[{index}]")


def validate_review(obj: object) -> dict[str, str]:
    """Check a review verdict against ``REVIEW_SCHEMA`` and return its fields."""
    check_schema(obj, REVIEW_SCHEMA["schema"])
    assert isinstance(obj, dict)  # narrowed by check_schema
    return {"classification": obj["classification"], "message": obj["message"]}


@dataclass(frozen=True)
class Review:
    """One reviewer's verdict.

    ``classification`` is a ``CLASSIFICATIONS`` member, or ``ENGAGE`` for a
    triager that wants the pipeline to continue. ``model`` records which
    reviewer produced it, for the combine stage and debug logs.
    """

    classification: str
    message: str = ""
    label_changes: tuple[JsonObject, ...] = ()
    model: str = ""

    @property
    def terminal(self) -> bool:
        return self.classification in TERMINAL_ROUTES

    @property
    def found_issue(self) -> bool:
        return self.classification in ISSUE_CLASSIFICATIONS


@dataclass
class ReviewContext:
    """The PR under review plus the drafts the pipeline accumulates.

    ``new_shell`` returns a fresh, isolated container shell (its own
    working trees) on each call; a reviewer that uses a shell owns it and
    closes it. ``None`` means no podman shell is configured (the OpenAI
    solo path uses its native tools instead). The pipeline -- not the
    reviewers -- appends to ``drafts``.
    """

    request: JsonObject
    patch_text: str
    patch_truncated: bool
    source_bundle: str | None
    source_files: list[str]
    source_notes: list[str]
    reviewer_username: str
    ci_triage_mode: bool
    repo_roots: list[Path]
    repo_mount_paths: list[str]
    new_shell: Callable[[], ContainerShellSession] | None = None
    drafts: list[Review] = field(default_factory=list)

    def review_drafts(self) -> list[Review]:
        """Prior drafts that are real review verdicts (excluding triage
        ``ENGAGE``), in the order produced -- what the combine stage merges."""
        return [d for d in self.drafts if d.classification in CLASSIFICATIONS]


class Reviewer(ABC):
    """The interface every model backend and the pipeline share.

    ``name`` is a short stable label (e.g. ``"openai:gpt-5.4"``) used in
    logs and recorded on the produced ``Review``.
    """

    name: str

    @abstractmethod
    def review(self, ctx: ReviewContext) -> Review: ...


def run_parallel(reviewers: list[Reviewer], ctx: ReviewContext) -> list[Review]:
    """Run each reviewer concurrently; return their drafts in input order.

    A reviewer that raises (provider outage, exhausted quota, ...) is
    dropped with a logged traceback so the surviving drafts still produce
    a review; only when every reviewer fails is the run aborted.

    Each reviewer obtains its own shell via ``ctx.new_shell`` so concurrent
    runs never share a container working tree. Results are gathered only
    after every thread finishes, so the shared ``ctx`` is never mutated
    concurrently; the caller extends ``ctx.drafts`` with the returned list.
    """
    if not reviewers:
        return []
    logger.info(
        "run_parallel: launching %d reviewers: %s",
        len(reviewers),
        ", ".join(r.name for r in reviewers),
    )
    with ThreadPoolExecutor(max_workers=len(reviewers)) as executor:
        futures = [executor.submit(r.review, ctx) for r in reviewers]
    drafts: list[Review] = []
    failed: list[str] = []
    for reviewer, future in zip(reviewers, futures):
        try:
            drafts.append(future.result())
        except Exception:
            failed.append(reviewer.name)
            logger.exception(
                "reviewer %s failed; continuing with the surviving drafts",
                reviewer.name,
            )
    if not drafts:
        raise RuntimeError(f"all reviewers failed: {', '.join(failed)}")
    return drafts
