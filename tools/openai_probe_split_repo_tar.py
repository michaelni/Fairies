#!/usr/bin/env python3
"""Build split plain tar archives for a repo and extract them sequentially.

This is a standalone debug tool for testing whether splitting a repository into
multiple plain `.tar` files and extracting them one by one inside an OpenAI
container works better than a single large archive.

The script:
- walks the repository tree
- partitions entries into multiple plain `.tar` parts
- keeps every part on disk
- uploads each part to OpenAI Files
- attaches each uploaded part to a fresh OpenAI container
- runs one shell probe that extracts the parts sequentially with verbose output

Nothing is deleted automatically. Local part files and metadata are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Probe tools live in tools/ but import from project-root modules
# (openai_common). Adjust sys.path so the script works regardless of
# whether it's invoked as ``python tools/<name>.py`` (sys.path[0] is
# tools/) or executed directly via the shebang.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from openai_common import load_api_key


DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_MEMORY_LIMIT = "1g"
DEFAULT_OUTPUT_DIRNAME = ".openai_container_debug"
DEFAULT_MAX_PART_BYTES = 200_000_000
DEFAULT_MAX_PART_MEMBERS = 20_000


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def emit(event: str, **payload: object) -> None:
    print(json.dumps({"ts": utc_timestamp(), "event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a repository into multiple plain .tar parts, upload them to an "
            "OpenAI container, and test sequential tar -xf extraction."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository checkout to archive. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRNAME,
        help="Directory where split .tar parts and metadata JSON are written.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run name. Defaults to <repo-name>-<head>-split-tar.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Responses model to use for the shell probe.",
    )
    parser.add_argument(
        "--memory-limit",
        choices=["1g", "4g", "16g", "64g"],
        default=DEFAULT_MEMORY_LIMIT,
        help="OpenAI container memory limit.",
    )
    parser.add_argument(
        "--expiry-minutes",
        type=int,
        default=20,
        help="Container expiry after last active time.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8000,
        help="Responses max_output_tokens for the shell probe.",
    )
    parser.add_argument(
        "--max-part-bytes",
        type=int,
        default=DEFAULT_MAX_PART_BYTES,
        help="Target max total regular-file bytes per tar part.",
    )
    parser.add_argument(
        "--max-part-members",
        type=int,
        default=DEFAULT_MAX_PART_MEMBERS,
        help="Target max member count per tar part.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Optional cap on total walked entries (0 = no limit).",
    )
    parser.add_argument(
        "--skip-dot-git",
        action="store_true",
        help="Skip entries under .git/.",
    )
    parser.add_argument(
        "--preview-members",
        type=int,
        default=20,
        help="How many member names to print from each part during extraction.",
    )
    return parser.parse_args()


def get_repo_head(repo_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return "unknown-head"
    if cp.returncode == 0:
        return cp.stdout.strip() or "unknown-head"
    return "unknown-head"


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_entry(repo_root: Path, path: Path) -> dict[str, object] | None:
    try:
        st = path.lstat()
    except OSError as exc:
        emit("walk.skip.stat_error", path=str(path), error=str(exc))
        return None
    relpath = str(path.relative_to(repo_root))
    mode = stat.S_IMODE(st.st_mode)
    if stat.S_ISDIR(st.st_mode):
        return {"entry_type": "dir", "repo_relpath": relpath, "local_path": str(path), "mode": mode, "size_bytes": 0}
    if stat.S_ISREG(st.st_mode):
        return {
            "entry_type": "file",
            "repo_relpath": relpath,
            "local_path": str(path),
            "mode": mode,
            "size_bytes": st.st_size,
        }
    if stat.S_ISLNK(st.st_mode):
        try:
            link_target = os.readlink(path)
        except OSError as exc:
            emit("walk.skip.readlink_error", path=str(path), error=str(exc))
            return None
        return {
            "entry_type": "symlink",
            "repo_relpath": relpath,
            "local_path": str(path),
            "mode": mode,
            "size_bytes": 0,
            "link_target": link_target,
        }
    emit("walk.skip.unsupported_entry", path=str(path), mode=oct(st.st_mode))
    return None


def walk_repo_entries(repo_root: Path, *, output_dir: Path, skip_dot_git: bool, max_entries: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(repo_root.rglob("*")):
        if path_is_within(path, output_dir):
            emit("walk.skip.output_dir", path=str(path), output_dir=str(output_dir))
            continue
        rel_parts = path.relative_to(repo_root).parts
        if skip_dot_git and ".git" in rel_parts:
            continue
        entry = build_entry(repo_root, path)
        if entry is None:
            continue
        entries.append(entry)
        if max_entries > 0 and len(entries) >= max_entries:
            emit("walk.max_entries_reached", max_entries=max_entries)
            break
    return entries


def partition_entries(entries: list[dict[str, object]], *, max_part_bytes: int, max_part_members: int) -> list[list[dict[str, object]]]:
    parts: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_bytes = 0
    current_members = 0
    for entry in entries:
        entry_bytes = int(entry.get("size_bytes", 0))
        would_exceed_bytes = current and max_part_bytes > 0 and current_bytes + entry_bytes > max_part_bytes
        would_exceed_members = current and max_part_members > 0 and current_members + 1 > max_part_members
        if would_exceed_bytes or would_exceed_members:
            parts.append(current)
            current = []
            current_bytes = 0
            current_members = 0
        current.append(entry)
        current_bytes += entry_bytes
        current_members += 1
    if current:
        parts.append(current)
    return parts


def build_part_tar(repo_root: Path, repo_name: str, part_entries: list[dict[str, object]], archive_path: Path) -> dict[str, object]:
    started = time.monotonic()
    emit(
        "archive.build.start",
        archive_path=str(archive_path),
        repo_root=str(repo_root),
        entry_count=len(part_entries),
        mode="w",
        format="PAX_FORMAT",
    )
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(repo_root, arcname=repo_name, recursive=False)
        for entry in part_entries:
            local_path = Path(str(entry["local_path"]))
            arcname = f"{repo_name}/{entry['repo_relpath']}"
            archive.add(local_path, arcname=arcname, recursive=False)
    result = {
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "entry_count": len(part_entries),
        "regular_file_bytes": sum(int(entry.get("size_bytes", 0)) for entry in part_entries),
        "preview": [str(entry["repo_relpath"]) for entry in part_entries[:10]],
        "dt_s": round(time.monotonic() - started, 3),
    }
    emit("archive.build.ok", **result)
    return result


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo root not found: {repo_root}")
    if not repo_root.is_dir():
        raise SystemExit(f"repo root is not a directory: {repo_root}")

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("missing OPENAI_API_KEY in environment or .env")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_name = repo_root.name
    head_sha = get_repo_head(repo_root)
    run_name = args.run_name or f"{repo_name}-{head_sha[:16]}-split-tar"
    metadata_path = output_dir / f"{run_name}.metadata.json"
    part_manifest_path = output_dir / f"{run_name}.parts.json"

    metadata: dict[str, object] = {
        "repo_root": str(repo_root),
        "repo_name": repo_name,
        "head_sha": head_sha,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "part_manifest_path": str(part_manifest_path),
        "model": args.model,
        "memory_limit": args.memory_limit,
        "expiry_minutes": args.expiry_minutes,
        "max_output_tokens": args.max_output_tokens,
        "max_part_bytes": args.max_part_bytes,
        "max_part_members": args.max_part_members,
        "max_entries": args.max_entries,
        "skip_dot_git": args.skip_dot_git,
        "preview_members": args.preview_members,
    }
    write_json(metadata_path, metadata)
    emit("metadata.write", metadata_path=str(metadata_path), part_manifest_path=str(part_manifest_path))

    walk_started = time.monotonic()
    entries = walk_repo_entries(
        repo_root,
        output_dir=output_dir,
        skip_dot_git=args.skip_dot_git,
        max_entries=args.max_entries,
    )
    metadata["walk_entry_count"] = len(entries)
    metadata["walk_file_count"] = sum(1 for entry in entries if entry["entry_type"] == "file")
    metadata["walk_dir_count"] = sum(1 for entry in entries if entry["entry_type"] == "dir")
    metadata["walk_symlink_count"] = sum(1 for entry in entries if entry["entry_type"] == "symlink")
    metadata["walk_total_file_bytes"] = sum(int(entry.get("size_bytes", 0)) for entry in entries)
    write_json(metadata_path, metadata)
    emit(
        "walk.ok",
        entry_count=metadata["walk_entry_count"],
        file_count=metadata["walk_file_count"],
        dir_count=metadata["walk_dir_count"],
        symlink_count=metadata["walk_symlink_count"],
        total_file_bytes=metadata["walk_total_file_bytes"],
        dt_s=round(time.monotonic() - walk_started, 3),
    )
    if not entries:
        emit("done", reason="no entries selected")
        return 0

    part_groups = partition_entries(
        entries,
        max_part_bytes=args.max_part_bytes,
        max_part_members=args.max_part_members,
    )
    emit("partition.ok", part_count=len(part_groups))

    part_records: list[dict[str, object]] = []
    for index, part_entries in enumerate(part_groups, start=1):
        archive_path = output_dir / f"{run_name}.part{index:03d}.tar"
        build_info = build_part_tar(repo_root, repo_name, part_entries, archive_path)
        record = {
            "index": index,
            "archive_path": str(archive_path),
            "entry_count": len(part_entries),
            "regular_file_bytes": sum(int(entry.get("size_bytes", 0)) for entry in part_entries),
            "entries_preview": [str(entry["repo_relpath"]) for entry in part_entries[:20]],
            **build_info,
        }
        part_records.append(record)
        emit("part.recorded", **record)

    part_manifest = {
        "generated_at": utc_timestamp(),
        "repo_root": str(repo_root),
        "repo_name": repo_name,
        "head_sha": head_sha,
        "parts": part_records,
    }
    write_json(part_manifest_path, part_manifest)
    metadata["part_count"] = len(part_records)
    metadata["parts"] = part_records
    write_json(metadata_path, metadata)
    emit("part_manifest.write", path=str(part_manifest_path), part_count=len(part_records))

    client = OpenAI(api_key=api_key)

    create_started = time.monotonic()
    emit("openai.containers.create.start", memory_limit=args.memory_limit, expiry_minutes=max(1, args.expiry_minutes))
    container = client.containers.create(
        name=f"debug split tar {repo_name}",
        memory_limit=args.memory_limit,
        expires_after={"anchor": "last_active_at", "minutes": max(1, args.expiry_minutes)},
    )
    metadata["container_id"] = container.id
    write_json(metadata_path, metadata)
    emit("openai.containers.create.ok", container_id=container.id, dt_s=round(time.monotonic() - create_started, 3))

    attached_paths: list[str] = []
    output_root = f"/mnt/data/out/{repo_name}"
    for record in part_records:
        archive_path = Path(str(record["archive_path"]))
        upload_started = time.monotonic()
        emit(
            "openai.files.create.start",
            part_index=record["index"],
            archive_path=str(archive_path),
            bytes=archive_path.stat().st_size,
            purpose="user_data",
        )
        with archive_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="user_data")
        record["uploaded_file_id"] = uploaded.id
        emit(
            "openai.files.create.ok",
            part_index=record["index"],
            file_id=uploaded.id,
            bytes=archive_path.stat().st_size,
            dt_s=round(time.monotonic() - upload_started, 3),
        )

        attach_started = time.monotonic()
        emit(
            "openai.containers.files.create.start",
            part_index=record["index"],
            container_id=container.id,
            file_id=uploaded.id,
        )
        container_file = client.containers.files.create(container_id=container.id, file_id=uploaded.id)
        record["container_file_path"] = container_file.path
        record["container_file_bytes"] = getattr(container_file, "bytes", None)
        attached_paths.append(container_file.path)
        write_json(part_manifest_path, part_manifest)
        write_json(metadata_path, metadata)
        emit(
            "openai.containers.files.create.ok",
            part_index=record["index"],
            container_id=container.id,
            file_id=uploaded.id,
            path=container_file.path,
            bytes=getattr(container_file, "bytes", None),
            dt_s=round(time.monotonic() - attach_started, 3),
        )
        quoted = json.dumps(container_file.path)
        command = (
            "set -euxo pipefail; "
            f"echo SPLIT_PART_START index={record['index']} path={quoted}; "
            "date +%s; "
            "pwd; "
            "python3 --version; "
            "df -h; "
            "ls -lh /mnt/data; "
            "mkdir -p /mnt/data/out; "
            f"ls -lh {quoted}; "
            f"tar -tf {quoted} | sed -n '1,{max(1, args.preview_members)}p'; "
            f"tar -xf {quoted} -C /mnt/data/out; "
            "date +%s; "
            "du -sh /mnt/data/out; "
            f"ls -ld {json.dumps(output_root)} {json.dumps(output_root + '/.git')} || true; "
            f"test -f {json.dumps(output_root + '/.git/HEAD')} && sed -n '1,5p' {json.dumps(output_root + '/.git/HEAD')} || true; "
            f"echo SPLIT_PART_DONE index={record['index']} path={quoted}"
        )
        record["shell_command"] = command
        write_json(part_manifest_path, part_manifest)
        metadata["last_shell_command"] = command
        write_json(metadata_path, metadata)
        emit(
            "openai.responses.create.start",
            part_index=record["index"],
            container_id=container.id,
            model=args.model,
            command=command,
        )

        payload = {
            "model": args.model,
            "tool_choice": "required",
            "max_output_tokens": args.max_output_tokens,
            "tools": [{
                "type": "shell",
                "environment": {"type": "container_reference", "container_id": container.id},
            }],
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": f"Run exactly this command once and return its output only: {command}",
                }],
            }],
        }

        shell_started = time.monotonic()
        try:
            response = client.responses.create(**payload)
        except Exception as exc:
            record["shell_ok"] = False
            record["shell_error"] = str(exc)
            record["shell_error_type"] = type(exc).__name__
            record["shell_status_code"] = getattr(exc, "status_code", None)
            record["shell_request_id"] = getattr(exc, "request_id", None)
            record["shell_dt_s"] = round(time.monotonic() - shell_started, 3)
            metadata["shell_ok"] = False
            metadata["shell_error"] = str(exc)
            metadata["shell_error_type"] = type(exc).__name__
            metadata["shell_status_code"] = getattr(exc, "status_code", None)
            metadata["shell_request_id"] = getattr(exc, "request_id", None)
            metadata["shell_dt_s"] = round(time.monotonic() - shell_started, 3)
            metadata["failed_part_index"] = record["index"]
            write_json(part_manifest_path, part_manifest)
            write_json(metadata_path, metadata)
            emit(
                "openai.responses.create.error",
                part_index=record["index"],
                error_type=type(exc).__name__,
                error=str(exc),
                status_code=getattr(exc, "status_code", None),
                request_id=getattr(exc, "request_id", None),
                dt_s=round(time.monotonic() - shell_started, 3),
            )
            raise

        record["shell_ok"] = True
        record["shell_dt_s"] = round(time.monotonic() - shell_started, 3)
        record["response_id"] = getattr(response, "id", None)
        record["output_text"] = getattr(response, "output_text", None)
        metadata["last_completed_part_index"] = record["index"]
        metadata["shell_ok"] = True
        metadata["shell_dt_s"] = round(time.monotonic() - shell_started, 3)
        metadata["response_id"] = getattr(response, "id", None)
        metadata["output_text"] = getattr(response, "output_text", None)
        write_json(part_manifest_path, part_manifest)
        write_json(metadata_path, metadata)
        emit(
            "openai.responses.create.ok",
            part_index=record["index"],
            response_id=getattr(response, "id", None),
            dt_s=round(time.monotonic() - shell_started, 3),
            output_text=getattr(response, "output_text", None),
        )

    metadata["attached_part_count"] = len(attached_paths)
    metadata["attached_part_paths"] = attached_paths
    write_json(metadata_path, metadata)
    emit(
        "done",
        metadata_path=str(metadata_path),
        part_manifest_path=str(part_manifest_path),
        container_id=container.id,
        part_count=len(part_records),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
