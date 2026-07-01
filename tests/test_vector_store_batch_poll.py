"""Batch poll doneness and per-batch incomplete-file reconcile.

Doneness regression: a just-created batch briefly reports
``status=in_progress`` with every ``file_counts`` field still 0. The poll
gated doneness on ``in_progress <= 0``, so it returned immediately with
``completed=0``, never confirming the files processed (observed: all 476
batch polls in one run returned the all-zero state in ~0.25s). Doneness
must key on a terminal batch ``status`` instead.

Reconcile behaviour: when a batch does not fully complete -- files failed,
were cancelled, or were still in_progress when the poll timed out -- the
code re-lists what actually completed (batch-scoped), deletes the
incomplete entries so they cannot linger as poison, and re-attaches the
rest up to ``DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS`` times.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_vector_store as vs  # noqa: E402


def _counts(**kw: int) -> SimpleNamespace:
    base = {"completed": 0, "failed": 0, "cancelled": 0, "in_progress": 0, "total": 0}
    base.update(kw)
    # The SDK returns ``file_counts`` as a ``FileCounts`` model object, not a
    # dict; mirror that so the poll's attribute reads are exercised.
    return SimpleNamespace(**base)


def _batch(status: str, **counts: int) -> SimpleNamespace:
    return SimpleNamespace(id="vsfb_test", status=status, file_counts=_counts(**counts))


class _SeqBatches:
    """Polls one batch through a fixed sequence of retrieve() results."""

    def __init__(self, retrieve_seq: list[SimpleNamespace]) -> None:
        self._seq = retrieve_seq
        self.retrieve_calls = 0

    def create(self, *, vector_store_id: str, files: list) -> SimpleNamespace:
        return SimpleNamespace(id="vsfb_test")

    def retrieve(self, *, batch_id: str, vector_store_id: str) -> SimpleNamespace:
        item = self._seq[min(self.retrieve_calls, len(self._seq) - 1)]
        self.retrieve_calls += 1
        return item


class _RoundBatches:
    """One terminal batch per create(); each round declares its terminal
    status, counts, and which file ids completed."""

    def __init__(self, rounds: list[dict]) -> None:
        self.rounds = rounds
        self.created_payloads: list[list] = []
        self.list_files_calls = 0

    def create(self, *, vector_store_id: str, files: list) -> SimpleNamespace:
        idx = len(self.created_payloads)
        self.created_payloads.append(files)
        return SimpleNamespace(id=f"vsfb_{idx}")

    def retrieve(self, *, batch_id: str, vector_store_id: str) -> SimpleNamespace:
        rnd = self.rounds[int(batch_id.split("_")[1])]
        return SimpleNamespace(id=batch_id, status=rnd.get("status", "completed"), file_counts=_counts(**rnd["counts"]))

    def list_files(self, *, batch_id: str, vector_store_id: str, filter: str, limit: int, after: str | None = None) -> SimpleNamespace:
        self.list_files_calls += 1
        rnd = self.rounds[int(batch_id.split("_")[1])]
        data = [SimpleNamespace(id=fid) for fid in rnd.get("completed", [])]
        return SimpleNamespace(data=data, has_more=False, last_id=None)


class _FakeFiles:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, *, vector_store_id: str, file_id: str) -> SimpleNamespace:
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)


def _round_client(rounds: list[dict]) -> tuple[SimpleNamespace, _RoundBatches, _FakeFiles]:
    batches = _RoundBatches(rounds)
    files = _FakeFiles()
    client = SimpleNamespace(vector_stores=SimpleNamespace(file_batches=batches, files=files))
    return client, batches, files


class BatchPollTerminalStatusTests(unittest.TestCase):
    def test_waits_past_transitional_all_zero_until_completed(self) -> None:
        seq = [
            _batch("in_progress"),                                   # all zero
            _batch("in_progress", completed=1, in_progress=1),       # working
            _batch("completed", completed=2),                        # terminal
        ]
        batches = _SeqBatches(seq)
        client = SimpleNamespace(vector_stores=SimpleNamespace(file_batches=batches))
        with patch.object(vs.time, "sleep"):
            result = vs.create_vector_store_batch_and_poll(
                client, "vs_x",
                [("file-1", "a.c", "sha1"), ("file-2", "b.c", "sha2")],
                verbose=False,
            )
        self.assertEqual(batches.retrieve_calls, 3)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["timed_out"], 0)


class BatchIncompleteReconcileTests(unittest.TestCase):
    def test_failed_file_is_deleted_and_reattached_until_success(self) -> None:
        rounds = [
            {"counts": {"completed": 1, "failed": 1}, "completed": ["file-a"]},
            {"counts": {"completed": 1}, "completed": ["file-b"]},
        ]
        client, batches, files = _round_client(rounds)
        result = vs.create_vector_store_batch_and_poll(
            client, "vs_x",
            [("file-a", "a.c", "sha-a"), ("file-b", "b.c", "sha-b")],
            verbose=False,
        )
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(files.deleted, ["file-b"])
        self.assertEqual(len(batches.created_payloads), 2)
        self.assertEqual([f["file_id"] for f in batches.created_payloads[1]], ["file-b"])

    def test_timed_out_in_progress_file_is_reconciled_and_reattached(self) -> None:
        rounds = [
            {"status": "in_progress", "counts": {"completed": 1, "in_progress": 1}, "completed": ["file-a"]},
            {"counts": {"completed": 1}, "completed": ["file-b"]},
        ]
        client, batches, files = _round_client(rounds)
        with patch.object(vs, "DEFAULT_VECTOR_STORE_FILE_BATCH_POLL_TIMEOUT_SECONDS", 0.0), patch.object(vs.time, "sleep"):
            result = vs.create_vector_store_batch_and_poll(
                client, "vs_x",
                [("file-a", "a.c", "sha-a"), ("file-b", "b.c", "sha-b")],
                verbose=False,
            )
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(files.deleted, ["file-b"])
        self.assertEqual([f["file_id"] for f in batches.created_payloads[1]], ["file-b"])

    def test_persistent_failures_give_up_after_max_attempts_and_clean_poison(self) -> None:
        rounds = [{"counts": {"failed": 1}, "completed": []}] * vs.DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS
        client, batches, files = _round_client(rounds)
        result = vs.create_vector_store_batch_and_poll(
            client, "vs_x", [("file-a", "a.c", "sha-a")], verbose=False,
        )
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["completed"], 0)
        self.assertEqual(len(batches.created_payloads), vs.DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS)
        self.assertEqual(files.deleted, ["file-a"] * vs.DEFAULT_VECTOR_STORE_FILE_BATCH_MAX_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
