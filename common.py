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
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from typing import TypeAlias

logger = logging.getLogger(__name__)


# JSON value type aliases shared by every module that touches gcli or
# OpenAI responses. Defined here (rather than in ``forge_gcli`` or
# ``openai_common``) so neither subsystem has to depend on the other
# just to agree on the shape of a ``dict[str, JsonValue]``. Both
# subsystems re-import these names; consumers may pick either entry
# point.
JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def response_to_debug_json(response: object) -> JsonObject:
    """Best-effort JSON view of any SDK response object (OpenAI and
    Anthropic models are pydantic and expose ``model_dump``)."""
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(response, dict):
        return response
    return {"repr": repr(response)}


def dump_response_debug_artifacts(
    response: object,
    response_kwargs: dict[str, object],
    *,
    wrapper_request: JsonObject | None = None,
    debug_dir: str,
    verbose: bool,
    conversation: str | None = None,
) -> str | None:
    """Dump one JSONL record pairing an LLM API response with the exact
    request kwargs that produced it (vendor-neutral); returns the file path.

    Without ``conversation`` a new ``<response id>.jsonl`` is created. Pass
    a previous call's return value as ``conversation`` to append follow-up
    tool rounds there, so a whole tool-use conversation lands in one file
    instead of one file per round.
    """
    try:
        payload = {
            "response": response_to_debug_json(response),
            "request": response_kwargs,
        }
        if wrapper_request is not None:
            payload["wrapper_request"] = wrapper_request
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        if conversation:
            out_path = Path(conversation)
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            if verbose:
                logger.debug("appended response debug dump to %s", out_path)
        else:
            response_json = payload["response"]
            response_id = response_json.get("id") if isinstance(response_json, dict) else None
            stem = response_id if isinstance(response_id, str) and response_id else f"response_{int(time.time())}"
            out_dir = Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.jsonl"
            out_path.write_text(line, encoding="utf-8")
            if verbose:
                logger.debug("wrote response debug dump to %s", out_path)
        return str(out_path)
    except Exception as exc:
        logger.warning("failed to write response debug dump: %s", exc)
        return None


class _ThreadPrefixFilter(logging.Filter):
    """Maps thread names to short log prefixes. A ``~tag`` suffix on the
    thread name (the side's owner/repo when several run in one process)
    is carried into the prefix, so interleaved pipeline lines stay
    attributable to their repo."""

    _PREFIXES = {
        "MainThread": "M ",
        "pr-prepare": "P ",
        "pr-llm": "L ",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        base, _, tag = record.threadName.partition("~")
        prefix = self._PREFIXES.get(base, "T ")
        record.thread_prefix = f"{prefix.rstrip()} {tag} " if tag else prefix
        return True


# ANSI color codes used by ``_ColorFormatter`` to make non-INFO messages
# stand out in interactive runs. DEBUG is rendered as dim gray so it
# recedes; WARNING/ERROR/CRITICAL are rendered bold so they pop. INFO
# is left uncolored as the visual baseline.
#
# DEBUG combines two attributes -- ``2`` (faint) and ``90`` (bright
# black, i.e. gray) -- because neither alone renders reliably on every
# terminal. ``2`` is widely ignored (xterm, many embedded terminals,
# VS Code's integrated terminal commonly drop it); ``90`` is supported
# universally as a 16-color foreground but on its own is just "gray",
# not particularly dim. With both, terminals that honor only one still
# get a visible reduction, and terminals that honor both get a clear
# dim-gray that visibly recedes from the default-foreground baseline.
# Without that combo a previous attempt with bare ``2`` was reported
# as visually indistinguishable from default white, defeating the
# whole point of dimming debug output.
_COLOR_RESET = "\x1b[0m"
_LEVEL_COLOR = {
    logging.DEBUG:    "\x1b[2;90m",   # dim gray
    logging.WARNING:  "\x1b[1;33m",   # bold yellow
    logging.ERROR:    "\x1b[1;31m",   # bold red
    logging.CRITICAL: "\x1b[1;31m",   # bold red
}


class _ColorFormatter(logging.Formatter):
    """Wraps the formatted line in an ANSI color sequence per level.

    Falls back to plain text for INFO and any unmapped level. ``setup_logging``
    only installs this formatter when stderr is a real TTY and ``NO_COLOR``
    is unset, so log redirection / piping / CI logs stay uncolored.
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = _LEVEL_COLOR.get(record.levelno)
        if color is None:
            return text
        return f"{color}{text}{_COLOR_RESET}"


def apply_config_file_defaults(
    parser: argparse.ArgumentParser, argv: list[str] | None = None,
) -> None:
    """Register ``--config FILE`` and load its TOML values as defaults.

    Keys are long option names (dashes or underscores interchangeably);
    each value becomes that option's default, so explicit command-line
    options win -- ``@argsfile`` semantics with comments and nicer
    syntax, nothing more. Unknown keys error; string values go through
    the option's ``type`` (command-line defaults bypass it otherwise).
    Call after every ``add_argument``, before ``parse_args``.
    """
    parser.add_argument(
        "--config", type=Path, metavar="FILE",
        help="TOML file of option-name = value defaults; explicit options win.",
    )
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    if known.config is None:
        return
    with open(known.config, "rb") as fh:
        cfg = tomllib.load(fh)
    actions = {a.dest: a for a in parser._actions}
    defaults: dict[str, object] = {}
    for key, value in cfg.items():
        dest = key.replace("-", "_")
        action = actions.get(dest)
        if action is None or dest == "help":
            parser.error(f"--config {known.config}: unknown option {key!r}")
        if action.type is not None:
            if isinstance(value, str):
                value = action.type(value)
            elif isinstance(value, list):
                value = [action.type(v) if isinstance(v, str) else v for v in value]
        defaults[dest] = value
    parser.set_defaults(**defaults)


def add_color_arg(parser: argparse.ArgumentParser) -> None:
    """Register ``--color={auto,always,never}`` on ``parser``.

    ``auto`` (default) enables color only when stderr is a real
    TTY and ``NO_COLOR`` is unset, OR when ``CLICOLOR_FORCE`` is
    set (so a single env var colors a whole pipeline of spawned
    subprocesses without needing ``--color always`` on every
    invocation). ``always`` forces color on even when stderr is a
    pipe -- useful with ``2>&1 | tee logfile`` or similar setups
    where the operator wants color in the live terminal and is
    fine with ANSI escapes leaking into the log file. ``never``
    disables color unconditionally and overrides ``CLICOLOR_FORCE``.
    """
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "Control ANSI color in log output (default: auto; "
            "CLICOLOR_FORCE forces color in auto mode)."
        ),
    )


def setup_logging(
    logger: logging.Logger,
    verbose: bool,
    *extra_loggers: logging.Logger,
    color: str = "auto",
    handlers: list[logging.Handler] | None = None,
) -> None:
    """Configure given loggers to print DEBUG/INFO/WARNING with timestamps.

    Attaches shared stdout/stderr handlers to ``logger``, each logger in
    ``extra_loggers``, and ``common``'s own module logger (this module
    hosts shared helpers that log, and it does know itself). The caller
    is responsible for listing the sibling module loggers it depends on
    -- ``common.py`` deliberately does NOT know the set of consumers, so
    the dependency direction stays one-way (top-level entry points know
    their helpers, the shared utility module does not know its users).

    Why multiple loggers instead of configuring the root logger: we
    would rather not eagerly route every third-party library's DEBUG
    traffic (e.g. ``openai``, ``httpx``) through our formatter when
    ``verbose=True``. Restricting handler attachment to a known set of
    project loggers keeps ``--verbose`` output scoped to our own code.

    Sharing handler instances across loggers is safe: ``logging.Handler``
    is designed for multi-logger attachment, and the filters we install
    are stateless. ``propagate`` is turned off on each target so
    messages are never emitted twice via the root logger's lastResort
    handler.
    """
    level = logging.DEBUG if verbose else logging.INFO

    if handlers is not None:
        # ``handlers`` replaces the stderr stream handlers entirely.
        thread_prefix_filter = _ThreadPrefixFilter()
        seen: set[int] = set()
        for target in (logger, *extra_loggers, logging.getLogger(__name__)):
            if id(target) in seen:
                continue
            seen.add(id(target))
            target.setLevel(level)
            target.handlers.clear()
            target.propagate = False
            for handler in handlers:
                handler.addFilter(thread_prefix_filter)
                target.addHandler(handler)
        return

    # ``auto`` (default) keeps color tied to a real TTY and respects the
    # de-facto NO_COLOR convention so CI logs and redirected runs stay
    # plain. ``always`` forces it on even when stderr is a pipe (useful
    # with ``2>&1 | tee logfile`` -- the live terminal stays colored at
    # the cost of ANSI escapes in the log file). ``never`` disables
    # color unconditionally.
    #
    # ``CLICOLOR_FORCE`` is the env-var counterpart to ``NO_COLOR`` (BSD /
    # fish / coreutils convention): when set it forces color on even
    # without a TTY. Honored only on the ``auto`` path -- explicit
    # ``--color=never`` still disables. The advantage over ``--color
    # always`` is that env vars are inherited by spawned subprocesses
    # (e.g. ``fairy.py`` -> ``pr_review_wrapper.py``),
    # so a single env-var setting colors the whole pipeline rather than
    # requiring ``--color always`` to be threaded through every command.
    if color == "always":
        use_color = True
    elif color == "never":
        use_color = False
    elif os.environ.get("CLICOLOR_FORCE"):
        use_color = True
    else:
        use_color = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    if os.environ.get("FAIRY_LOG_WIRE"):
        # a piped wrapper logs in add_file_log's exact shape so the
        # relaying pump can recover level and time; color and thread
        # prefixes would only litter the wire
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname).1s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S',
        )
    else:
        formatter_cls = _ColorFormatter if use_color else logging.Formatter
        formatter = formatter_cls(
            fmt='%(asctime)s %(thread_prefix)s%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    thread_prefix_filter = _ThreadPrefixFilter()

    # INFO goes to stderr (not stdout) because several scripts that use
    # this logging setup also write machine-readable data to stdout
    # (``pr_review_wrapper.py`` writes the review-decision JSON,
    # ``--prepare-vector-store-only`` writes a vector-store dump). Any
    # INFO line on stdout would otherwise be concatenated with that data
    # and break the parent's JSON parse (e.g. the leading "2026" in a
    # timestamp parses as a JSON number, turning the captured stdout
    # into a multi-value list).
    # info_handler.setLevel(INFO) accepts INFO and above, so without an
    # explicit "INFO only" filter every WARNING/ERROR would print twice
    # (once via info_handler, once via warn_handler). The lambda is the
    # de-dup gate.
    info_handler = logging.StreamHandler(sys.stderr)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(thread_prefix_filter)
    info_handler.addFilter(lambda r: r.levelno == logging.INFO)
    info_handler.setFormatter(formatter)

    # No level-filter lambda needed: setLevel(WARNING) already drops
    # everything below WARNING.
    warn_handler = logging.StreamHandler(sys.stderr)
    warn_handler.setLevel(logging.WARNING)
    warn_handler.addFilter(thread_prefix_filter)
    warn_handler.setFormatter(formatter)

    # Same de-dup reason as info_handler: setLevel(DEBUG) accepts every
    # level, so without "DEBUG only" each INFO/WARNING would print twice.
    debug_handler: logging.Handler | None = None
    if verbose:
        debug_handler = logging.StreamHandler(sys.stderr)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(thread_prefix_filter)
        debug_handler.addFilter(lambda r: r.levelno == logging.DEBUG)
        debug_handler.setFormatter(formatter)

    # De-dupe by identity so passing the same logger twice is a no-op
    # rather than attaching duplicate handlers.
    seen: set[int] = set()
    for target in (logger, *extra_loggers, logging.getLogger(__name__)):
        if id(target) in seen:
            continue
        seen.add(id(target))
        target.setLevel(level)
        target.handlers.clear()
        target.propagate = False
        target.addHandler(info_handler)
        target.addHandler(warn_handler)
        if debug_handler is not None:
            target.addHandler(debug_handler)


def watch_paths(paths: list[Path], callback) -> object | None:
    """Fire ``callback()`` (from the observer thread; keep it to setting
    an Event) on any change under the given directories. Returns the
    started watchdog observer, or None when the watchdog package is not
    installed or a path cannot be watched -- callers keep their
    interval fallback and merely react slower."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        return None

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event) -> None:
            callback()

    observer = Observer()
    observer.daemon = True
    handler = _Handler()
    for path in paths:
        try:
            observer.schedule(handler, str(path))
        except OSError as exc:
            logging.getLogger(__name__).debug("cannot watch %s: %s", path, exc)
    observer.start()
    return observer


def add_file_log(path: Path, logger: logging.Logger,
                 *extra_loggers: logging.Logger) -> None:
    """Additionally log to ``path`` in the fixed ``ISO8601 L message``
    shape (single-letter level) that the fairy-ui tail pane parses to
    color merged agent/worker logs by level."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname).1s %(message)s', '%Y-%m-%dT%H:%M:%S'))
    seen: set[int] = set()
    for target in (logger, *extra_loggers, logging.getLogger(__name__)):
        if id(target) not in seen:
            seen.add(id(target))
            target.addHandler(handler)


_REPO_NAME_INVALID_RE = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_repo_name(raw: str) -> str:
    """Map an arbitrary directory name onto a safe in-container identifier.

    Non-alphanumerics (other than ``._-``) collapse to ``-``; leading/
    trailing punctuation is trimmed. Empty results fall back to
    ``"repo"`` so callers always get a usable string.
    """
    return _REPO_NAME_INVALID_RE.sub("-", raw).strip("._-") or "repo"


def dedup_with_suffix(name: str, used: set[str]) -> str:
    """Return ``name`` if unused, else ``f"{name}-{N}"`` for the lowest free N>=2.

    Mutates ``used`` to record the chosen result so subsequent calls
    avoid the same value.
    """
    if name not in used:
        used.add(name)
        return name
    suffix = 2
    while f"{name}-{suffix}" in used:
        suffix += 1
    chosen = f"{name}-{suffix}"
    used.add(chosen)
    return chosen


def default_cache_path(filename: str) -> Path:
    return Path.home() / ".fairy" / filename


def attachment_urls(obj: JsonObject) -> list[JsonObject]:
    """Forgejo/Gitea ``assets`` of an issue or comment; an unlinked
    attachment appears nowhere in the markdown body. GitHub and GitLab
    upload as inline body links and have no such field."""
    return [
        {"name": a.get("name"), "size": a.get("size"),
         "url": a.get("browser_download_url")}
        for a in obj.get("assets") or []
    ]


def iso_to_dt(value: str | None) -> datetime | None:
    """Parse a Forgejo/Gitea ISO-8601 timestamp into a UTC ``datetime``.

    Forgejo emits timestamps with both ``Z`` and explicit ``+HH:MM``
    suffixes; the leading ``Z`` is folded to ``+00:00`` so the standard
    ``datetime.fromisoformat`` parser accepts it on every supported
    Python version. Naive results (no tzinfo) are assumed to be UTC.
    Returns ``None`` for empty / unparseable input -- callers that
    sort/compare timestamps already filter ``None`` separately.
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_iso_datetime_arg(value: str) -> datetime:
    """``argparse`` ``type=`` adapter around :func:`iso_to_dt`."""
    dt = iso_to_dt(value)
    if dt is None:
        raise argparse.ArgumentTypeError(f"invalid ISO date/time: {value!r}")
    return dt


def atomic_write_pickle(path: Path, obj: object) -> None:
    """Pickle ``obj`` to ``path`` atomically (mkstemp -> dump -> os.replace).

    Companion to ``atomic_write_text``. Unlike that helper this one
    deliberately does NOT fsync before the rename: the existing
    pickle-based caches (fairy's discussion cache,
    mail_fairy's forwarded-msgid state) treat themselves as advisory
    -- a torn write at process kill survives a fresh start because
    the loaders fall back to an empty cache on any deserialization
    failure. Adding fsync would only buy crash-consistency at the
    cost of write throughput, and is intentionally left as a future
    decision.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically: write to a temp sibling file,
    fsync, then rename into place.

    Safe for concurrent callers racing on the same ``path``: the last
    ``os.replace`` wins and all readers observe either the previous file or
    the new one, never a partially-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
