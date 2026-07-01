"""Persistent local state for ``fairy``'s LLM-skip backoff.

Pickle store keyed by ``(owner, repo, number)``. Each entry tracks
the most recent LLM verdict so a repeated "skip" on the same head
SHA + last_activity does not keep re-calling the LLM. The store is
informational only -- losing it costs one extra LLM call per
previously-suppressed PR -- so a wrong-version / unreadable pickle
is silently cold-warmed.

Does NOT store forge-fetched data: that belongs to ``gcli_cache``.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

from common import atomic_write_pickle

__all__ = ["Key", "Entry", "State", "SCHEMA_VERSION", "load", "save", "logger"]

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


class Key(NamedTuple):
    owner: str
    repo: str
    number: int


class Entry(TypedDict, total=False):
    last_llm_decision: str                  # raw classification string
    last_llm_at: str                        # ISO-8601 of the LLM call
    last_llm_head_sha: str | None           # PR head SHA at call time
    last_llm_last_activity_iso: str | None  # ISO of discussion last_activity at call time
    consecutive_skip_count: int             # 0 if last decision was non-skip


@dataclass
class State:
    version: int = SCHEMA_VERSION
    entries: dict[Key, Entry] = field(default_factory=dict)


def load(path: Path) -> State:
    try:
        obj = pickle.loads(path.read_bytes())
    except FileNotFoundError:
        return State()
    except Exception as exc:
        logger.warning("bot_state load %s: %s; empty state", path, exc)
        return State()
    if isinstance(obj, State) and obj.version == SCHEMA_VERSION:
        logger.debug("bot_state load %s: %d entries", path, len(obj.entries))
        return obj
    logger.info("bot_state load %s: wrong shape/version; empty state", path)
    return State()


def save(path: Path, state: State) -> None:
    atomic_write_pickle(path, state)
    logger.debug("bot_state save %s: %d entries", path, len(state.entries))
