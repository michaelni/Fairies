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

Vendor-agnostic podman container management for the LLM review wrapper.

This module knows about Podman, the host, and the isolated network. It
deliberately does NOT know anything about OpenAI, Anthropic, or any
specific LLM provider's tool-call wire format -- that translation lives
in the per-vendor wrapper. The same primitives are intended to back an
Anthropic wrapper later without modification.

The host-side LAN-block (nftables) that prevents a compromised
container from reaching RFC1918/link-local destinations is set up
out-of-band by ``containers/setup_host`` and is NOT enforced inside
the container or by this module. The intent is that even a fully
compromised container cannot bypass the egress filter.

podman always runs on the ssh host: every control call is
``ssh DEST podman ...`` via ``RemoteHost`` (that account may be local, a
VM, or in the cloud -- it makes no difference). Fixed-shape control
commands are shlex-joined so the remote shell re-tokenizes them to the
exact argv. The LLM's open-ended shell, which cannot be safely joined
that way, never touches ssh argv: it flows through a single persistent
``ContainerShellSession`` -- one ``ssh DEST podman exec -i <cid> python3
<agent>`` pipe driving ``containers/fairy_agent.py``, so commands cross
as JSON data with no re-tokenisation and no per-command handshake.

This replaces the former ``podman --remote`` control plane: plain ssh
authenticates with OpenSSH, needs no ``podman.socket``/linger on the
host, and keeps the podman REST API off the local box entirely.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteHost:
    """The ssh account where podman runs and the containers live.

    Podman is *always* reached as ``ssh host podman ...`` (that account
    may be local, a VM, or in the cloud -- it makes no difference here).
    Plain ssh is used on purpose instead of ``podman --remote``: it
    authenticates with OpenSSH (config/agent/default keys, no Go-ssh
    quirks), needs no ``podman.socket``/linger on the host, and keeps the
    podman REST API off the local box entirely.

    Fixed-shape control commands are shlex-joined into a single remote
    command string, so the remote login shell re-tokenizes them back to
    the exact argv. Only commands assembled in this codebase may go
    through here -- never untrusted or LLM-authored shell, which runs in
    the container via the ``ContainerShellSession`` agent protocol.
    """

    ssh_dest: str
    ssh_opts: tuple[str, ...] = (
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        # The codex relay reads its readiness marker off this stderr; an
        # ssh warning line there would masquerade as a failed relay start.
        "-o", "LogLevel=ERROR",
    )
    identity: str | None = None
    port: int | None = None

    def ssh_argv(self) -> list[str]:
        """``ssh`` plus client options, without the destination -- also
        usable verbatim as ``GIT_SSH_COMMAND``."""
        return ["ssh", *self.ssh_opts,
                *(["-i", self.identity] if self.identity else []),
                *(["-p", str(self.port)] if self.port else [])]

    def argv(self, remote_argv: Sequence[str]) -> list[str]:
        return [*self.ssh_argv(), self.ssh_dest, shlex.join(remote_argv)]


@dataclass(frozen=True)
class ContainerHandle:
    container_id: str
    image: str
    network: str | None
    host: RemoteHost


@dataclass(frozen=True)
class ExecResult:
    """Outcome of a single shell command executed inside the container.

    ``exit_code`` follows the agent's ``timeout(1)`` convention: ``124``
    means the agent watchdog killed the command's process group, ``127``
    means the command could not be launched. A dead channel (agent
    unresponsive past the host margin, or ssh dropped) is not an
    ``ExecResult`` -- ``ContainerShellSession.exec`` raises instead, so
    the caller aborts the (ephemeral) review.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    stdout_truncated: bool
    stderr_truncated: bool


HOST_RESPONSE_MARGIN_S = 30.0

# The legitimate agent has at most one response in flight; blocking the
# reader pushes back on the pipe instead of growing host memory.
_RESPONSE_QUEUE_MAX = 32

# Reader-thread sentinel for a frame exceeding the line cap.
_OVERSIZED_FRAME = object()


class ContainerShellSession:
    """A persistent shell channel into a review container.

    Holds one long-lived process -- in production ``ssh host podman exec
    -i <cid> python3 <agent>`` -- and speaks the newline-delimited JSON
    protocol of ``containers/fairy_agent.py``. Each :meth:`exec` writes one
    request and reads its response, so the whole LLM session pays a single
    ssh handshake instead of one per command, and commands cross as data
    (no shell re-tokenisation, and no podman REST surface on the local
    side -- only this fixed JSON schema does).

    The response stream is untrusted: PR-derived code runs in the
    container and can write to the agent's stdout (e.g. via
    ``/proc/<agent>/fd/1``), so frames may be arbitrary hostile bytes,
    not just what ``fairy_agent`` emits. Every frame is therefore
    size-capped and type-checked here, and any malformed frame kills the
    channel (fail closed) rather than being interpreted.

    Constructed from the launch ``argv`` so tests can run the real agent
    directly (``[python3, fairy_agent.py]``) without podman.
    """

    def __init__(self, launch_argv: Sequence[str], *,
                 max_output_bytes: int = 256 * 1024) -> None:
        self._argv = list(launch_argv)
        self._max_output_bytes = max_output_bytes
        # A well-formed frame holds two base64 payloads of at most
        # max_output_bytes each (4/3 expansion) plus small fixed fields.
        self._max_frame_bytes = 3 * max_output_bytes + 65536
        self._lock = threading.Lock()
        self._next_id = 0
        self._proc: subprocess.Popen | None = None
        self._responses: queue.Queue = queue.Queue(maxsize=_RESPONSE_QUEUE_MAX)

    def start(self) -> "ContainerShellSession":
        logger.info("opening container shell session cmd=%s", shlex.join(self._argv))
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        return self

    def _read_loop(self) -> None:
        # readline (not iteration) so a flushed line is delivered at once
        # rather than waiting for the iterator's read-ahead buffer to fill.
        stdout = self._proc.stdout
        try:
            while True:
                line = stdout.readline(self._max_frame_bytes)
                if not line:
                    return
                if len(line) >= self._max_frame_bytes and not line.endswith(b"\n"):
                    self._responses.put(_OVERSIZED_FRAME)
                    return
                self._responses.put(line)
        finally:
            self._responses.put(None)  # EOF sentinel

    def _drain_stderr(self) -> None:
        stderr = self._proc.stderr
        while True:
            line = stderr.readline()
            if not line:
                return
            logger.debug("container shell stderr: %s",
                         line.decode("utf-8", "replace").rstrip())

    def exec(self, command: str, *, cwd: str | None = None,
             timeout_s: float = 120.0) -> ExecResult:
        if self._proc is None:
            raise RuntimeError("container shell session not started")
        with self._lock:
            return self._exec_locked(command, cwd, timeout_s)

    def _exec_locked(self, command: str, cwd: str | None,
                     timeout_s: float) -> ExecResult:
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"container shell channel closed (exit={self._proc.returncode})"
            )
        req_id = self._next_id
        self._next_id += 1
        req = {
            "id": req_id, "command": command, "cwd": cwd,
            "timeout_s": timeout_s, "max_output_bytes": self._max_output_bytes,
        }
        logger.debug("container shell req id=%d timeout=%.1fs cwd=%s cmd=%s",
                     req_id, timeout_s, cwd, command)
        t0 = time.monotonic()
        try:
            self._proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._terminate()
            raise RuntimeError("container shell channel closed (write failed)") from exc
        try:
            line = self._responses.get(timeout=timeout_s + HOST_RESPONSE_MARGIN_S)
        except queue.Empty:
            logger.error(
                "container shell unresponsive id=%d timeout=%.1fs; killing channel",
                req_id, timeout_s,
            )
            self._terminate()
            raise RuntimeError("container shell timed out waiting for response")
        if line is None:
            self._terminate()
            raise RuntimeError("container shell channel closed (eof)")
        if line is _OVERSIZED_FRAME:
            self._terminate()
            raise RuntimeError(
                f"container shell protocol violation: frame exceeds "
                f"{self._max_frame_bytes} bytes"
            )
        # ValueError covers JSONDecodeError, UnicodeDecodeError and binascii.Error.
        try:
            resp = json.loads(line)
        except ValueError as exc:
            self._terminate()
            raise RuntimeError(
                "container shell protocol violation: undecodable frame"
            ) from exc
        if not isinstance(resp, dict):
            self._terminate()
            raise RuntimeError(
                "container shell protocol violation: frame is not an object"
            )
        if resp.get("id") != req_id:
            self._terminate()
            raise RuntimeError(
                f"container shell protocol desync: want id={req_id} got {resp.get('id')!r}"
            )
        dt = time.monotonic() - t0
        try:
            exit_code = int(resp["exit_code"])
            stdout = base64.b64decode(resp["stdout_b64"]).decode("utf-8", "replace")
            stderr = base64.b64decode(resp["stderr_b64"]).decode("utf-8", "replace")
            duration_s = float(resp.get("duration_s", dt))
            stdout_truncated = bool(resp["stdout_truncated"])
            stderr_truncated = bool(resp["stderr_truncated"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._terminate()
            raise RuntimeError(
                f"container shell protocol violation: bad response field: {exc!r}"
            ) from exc
        if not math.isfinite(duration_s):
            # NaN/Infinity parse as JSON here but do not survive re-encoding
            # to strict JSON downstream (tool-result payloads).
            duration_s = dt
        result = ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration_s,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        logger.debug(
            "container shell resp id=%d rc=%d dt=%.3fs out=%d%s err=%d%s",
            req_id, result.exit_code, result.duration_s,
            len(result.stdout), " (trunc)" if result.stdout_truncated else "",
            len(result.stderr), " (trunc)" if result.stderr_truncated else "",
        )
        return result

    def _terminate(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        if self._proc is None:
            return
        logger.debug("closing container shell session")
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._terminate()
        for stream in (self._proc.stdout, self._proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass
        self._proc = None


@dataclass
class _CmdResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def _run_capture(cmd: list[str], *, timeout_s: float, label: str) -> _CmdResult:
    """Run ``cmd`` to completion capturing output, with debug timing logs.

    ``label`` tags the logs (e.g. ``"podman"`` / ``"ssh"``) so the debug
    stream stays greppable per the debug-visibility rule.
    """
    logger.debug("%s start cmd=%s timeout=%.1fs", label, shlex.join(cmd), timeout_s)
    t0 = time.monotonic()
    try:
        cp = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        logger.error("%s timeout cmd=%s timeout=%.1fs", label, shlex.join(cmd), timeout_s)
        raise
    dt = time.monotonic() - t0
    logger.debug(
        "%s done cmd=%s rc=%d dt=%.3fs out=%d err=%d",
        label, shlex.join(cmd), cp.returncode, dt, len(cp.stdout), len(cp.stderr),
    )
    return _CmdResult(returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)


def run_on_remote_host(
    host: RemoteHost,
    *remote_argv: str,
    timeout_s: float = 120.0,
) -> _CmdResult:
    """Run a fixed-shape control command on the podman host over ssh.

    See ``RemoteHost`` for why this exists and the strict limits on
    what may be passed through it.
    """
    return _run_capture(host.argv(remote_argv), timeout_s=timeout_s, label="ssh")


def _podman(host: RemoteHost, *args: str, timeout_s: float = 60.0) -> _CmdResult:
    """Run a short, bounded-output ``podman`` subcommand on the host over
    ssh (``image exists``, ``network create``, ``run -d``, ``rm -f``).
    The LLM's open-ended shell work goes through ``ContainerShellSession``
    instead."""
    return run_on_remote_host(host, "podman", *args, timeout_s=timeout_s)


def image_tag_exists(image_tag: str, *, host: RemoteHost) -> bool:
    """Return True if ``podman image exists <tag>`` succeeds."""
    return _podman(host, "image", "exists", image_tag).returncode == 0


def ensure_isolated_network(name: str, *, host: RemoteHost) -> None:
    """Create the named podman network if missing.

    The host nftables LAN-block that gives this network its "internet
    yes, LAN no" property is installed by ``containers/setup_host`` and
    is intentionally NOT touched here.
    """
    if _podman(host, "network", "exists", name).returncode == 0:
        logger.debug("isolated network already exists name=%s", name)
        return
    logger.info("creating isolated podman network name=%s", name)
    cp = _podman(host, "network", "create", name)
    if cp.returncode != 0:
        raise RuntimeError(
            f"podman network create {name!r} failed: {cp.stderr.decode(errors='replace').strip()}"
        )


def _build_over_ssh(
    host: RemoteHost,
    *,
    image_tag: str,
    dockerfile: Path,
    context_dir: Path,
    label_args: list[str],
    timeout_s: float,
) -> int:
    """Build the image on the host, streaming the context as a tar into
    ``podman build ... -`` over ssh stdin.

    stdout/stderr are inherited so apt-get progress is visible live, per
    the debug-visibility rule -- a capture-and-print-on-failure design
    would hide minutes of build output until after it failed.
    """
    try:
        rel_dockerfile = str(dockerfile.relative_to(context_dir))
    except ValueError as exc:
        raise RuntimeError(
            f"dockerfile {dockerfile} must live inside build context {context_dir}"
        ) from exc
    tar_argv = ["tar", "-C", str(context_dir), "-cf", "-", "."]
    build_argv = host.argv([
        "podman", "build", "-t", image_tag, *label_args, "-f", rel_dockerfile, "-",
    ])
    logger.info(
        "podman build over ssh: %s | %s timeout=%.1fs",
        shlex.join(tar_argv), shlex.join(build_argv), timeout_s,
    )
    t0 = time.monotonic()
    tar = subprocess.Popen(tar_argv, stdout=subprocess.PIPE)
    try:
        build = subprocess.Popen(build_argv, stdin=tar.stdout)
    except OSError:
        tar.stdout.close()
        tar.kill()
        tar.wait()
        raise
    tar.stdout.close()  # build now owns the read end
    try:
        rc = build.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.error("podman build over ssh timed out after %.1fs", timeout_s)
        build.kill()
        build.wait()
        tar.kill()
        tar.wait()
        raise
    tar.wait()
    logger.info("podman build over ssh done rc=%d dt=%.3fs", rc, time.monotonic() - t0)
    return rc


def image_label(image_tag: str, label_key: str, *, host: RemoteHost) -> str | None:
    """Return the value of ``label_key`` on ``image_tag``, or ``None`` if
    the image or the label is absent."""
    fmt = '{{ index .Config.Labels "' + label_key + '" }}'
    cp = _podman(host, "image", "inspect", "--format", fmt, image_tag)
    if cp.returncode != 0:
        return None
    value = cp.stdout.decode(errors="replace").strip()
    return value or None


def build_image_if_needed(
    *,
    image_tag: str,
    dockerfile: Path,
    context_dir: Path,
    force: bool = False,
    labels: dict[str, str] | None = None,
    host: RemoteHost,
    build_timeout_s: float = 3600.0,
) -> None:
    if not force and image_tag_exists(image_tag, host=host):
        logger.info("image already present tag=%s", image_tag)
        return
    label_args: list[str] = []
    for key, value in (labels or {}).items():
        label_args += ["--label", f"{key}={value}"]
    logger.info(
        "building container image tag=%s dockerfile=%s context=%s labels=%s",
        image_tag, dockerfile, context_dir, labels or {},
    )
    rc = _build_over_ssh(
        host, image_tag=image_tag, dockerfile=dockerfile,
        context_dir=context_dir, label_args=label_args, timeout_s=build_timeout_s,
    )
    if rc != 0:
        raise RuntimeError(
            f"podman build {image_tag!r} failed (rc={rc}); see streamed output above"
        )


# Resource limits for ephemeral review containers. The prompt advertises
# these to the model (llm_prompt), so change them here, not at call sites.
CONTAINER_MEMORY = "8g"
CONTAINER_CPUS = "8"
# Fork-bomb backstop. Verified 2026-08-09 on the fairy-review image that
# `apt-get install`, `make -j32`, and 300 concurrent processes all stay
# under this cap, so it bounds a hostile PR's process count without
# constraining real builds/fuzzing.
CONTAINER_PIDS_LIMIT = 4096


@dataclass(frozen=True)
class ShellHostSpec:
    """One machine that runs review containers, as configured on the CLI.

    ``label`` is the deployment-chosen machine name the LLM passes as the
    shell tool's ``machine`` parameter; the core never interprets it.
    """
    label: str
    host: RemoteHost
    cpus: str = CONTAINER_CPUS
    memory: str = CONTAINER_MEMORY
    gpu: str | None = None  # podman --device value, e.g. nvidia.com/gpu=0


def parse_shell_host(spec: str, *, identity: str | None = None) -> ShellHostSpec:
    """Parse ``[LABEL=]SSH_DEST[,port=N][,cpus=N][,memory=SIZE][,gpu=DEVICE]``.

    A bare ``user@host`` (or ssh alias) gets the label ``x86_64``.
    Raises ValueError on unknown keys; callers turn that into a CLI error.
    """
    first, *rest = spec.split(",")
    label = "x86_64"
    ssh_dest = first
    if "=" in first.split("@", 1)[0]:
        label, ssh_dest = first.split("=", 1)
    options: dict[str, str] = {}
    for segment in rest:
        key, sep, value = segment.partition("=")
        if not sep or key not in ("port", "cpus", "memory", "gpu"):
            raise ValueError(f"unknown key {key!r} in shell host spec {spec!r} "
                             "(valid: port=, cpus=, memory=, gpu=)")
        options[key] = value
    return ShellHostSpec(
        label=label,
        host=RemoteHost(ssh_dest, identity=identity,
                        port=int(options["port"]) if "port" in options else None),
        cpus=options.get("cpus", CONTAINER_CPUS),
        memory=options.get("memory", CONTAINER_MEMORY),
        gpu=options.get("gpu"),
    )


def start_ephemeral_container(
    *,
    image: str,
    host: RemoteHost,
    network: str | None = None,
    memory: str = CONTAINER_MEMORY,
    cpus: str = CONTAINER_CPUS,
    extra_args: tuple[str, ...] = (),
) -> ContainerHandle:
    """Start a fresh container that lives only for one review.

    The container runs ``sleep infinity`` as PID 1; the LLM's commands
    are issued through a ``ContainerShellSession``. ``--rm`` ensures the
    container is removed when stopped (or when the host process dies), so
    leftover state cannot be reused across reviews -- per the "no reuse"
    requirement.

    ``network`` is the podman network to attach. Pass ``None`` or ``""``
    to omit ``--network`` entirely and use podman's default (rootless
    default is pasta), which is the mode used until the out-of-band
    egress LAN-block is in place.

    The working directory comes from the image's ``WORKDIR``; we do NOT
    pass ``podman run -w``. podman 4.9.3 (rootless, overlay) rejects
    ``--workdir`` pointing at a directory created by a Containerfile
    ``WORKDIR`` instruction ("workdir ... does not exist on container")
    even though it exists and ``podman exec -w`` resolves it fine -- so
    the flag would only break startup without buying anything.
    """
    args = [
        "run", "-d", "--rm",
        *([f"--network={network}"] if network else []),
        f"--memory={memory}",
        f"--cpus={cpus}",
        f"--pids-limit={CONTAINER_PIDS_LIMIT}",
        # PR code runs as root inside; block it from re-gaining privileges
        # through a setuid binary. Reviews never need this (the image ships
        # no sudo and apt runs as the container's own root).
        "--security-opt=no-new-privileges",
        *extra_args,
        image, "sleep", "infinity",
    ]
    logger.info(
        "starting ephemeral container image=%s network=%s memory=%s cpus=%s extra_args=%s",
        image, network or "(default)", memory, cpus, list(extra_args),
    )
    cp = _podman(host, *args, timeout_s=120.0)
    if cp.returncode != 0:
        raise RuntimeError(
            f"podman run failed: {cp.stderr.decode(errors='replace').strip()}"
        )
    container_id = cp.stdout.decode(errors="replace").strip()
    if not container_id:
        raise RuntimeError("podman run returned empty container id")
    logger.info(
        "ephemeral container started id=%s image=%s network=%s",
        container_id[:12], image, network,
    )
    return ContainerHandle(
        container_id=container_id, image=image, network=network, host=host,
    )


def reap_stale_containers(
    host: RemoteHost, *, images: Sequence[str], older_than: str = "60m",
) -> int:
    """Force-remove leaked review/codex containers on ``host``.

    Fairy's containers run ``sleep infinity``, so an interrupted or crashed
    run (before the wrapper's cleanup) leaves them behind. This reaps only
    the given ``images`` and only containers in ``created``/``exited``
    state -- never ``running`` (a live review is ``Up``) and never
    ``paused`` (the poison path keeps those for forensics) -- and only
    those older than ``older_than`` (a podman duration), so a container a
    concurrent run just created is never removed. Best-effort; returns the
    number removed and logs on failure.
    """
    removed = 0
    for image in images:
        listed = _podman(
            host, "ps", "-aq",
            "--filter", f"ancestor={image}",
            "--filter", f"until={older_than}",
            "--filter", "status=created",
            "--filter", "status=exited",
            timeout_s=60.0,
        )
        if listed.returncode != 0:
            logger.warning(
                "reap: listing %s on %s failed: %s", image, host.ssh_dest,
                listed.stderr.decode(errors="replace").strip())
            continue
        ids = listed.stdout.decode(errors="replace").split()
        if not ids:
            continue
        rm = _podman(host, "rm", "-f", *ids, timeout_s=120.0)
        if rm.returncode == 0:
            removed += len(ids)
            logger.info(
                "reaped %d stale container(s) image=%s host=%s",
                len(ids), image, host.ssh_dest)
        else:
            logger.warning(
                "reap: rm of %d %s container(s) on %s failed: %s",
                len(ids), image, host.ssh_dest,
                rm.stderr.decode(errors="replace").strip())
    return removed


def pause_container(handle: ContainerHandle) -> None:
    """``podman pause`` the container, preserving it for forensics.

    Unlike :func:`stop_container` this leaves the container on the host --
    frozen, not removed -- so a review container suspected of tampering
    (see the codex poison path) can be inspected later. Best-effort:
    logs and returns on failure. The operator must ``podman rm -f`` it by
    hand once done, since nothing else will reclaim it.
    """
    logger.debug("pausing container id=%s", handle.container_id[:12])
    cp = _podman(handle.host, "pause", handle.container_id, timeout_s=60.0)
    if cp.returncode != 0:
        logger.warning(
            "podman pause id=%s failed: %s",
            handle.container_id[:12], cp.stderr.decode(errors="replace").strip(),
        )
        return
    logger.warning(
        "container PAUSED for forensics id=%s host=%s; inspect it, then "
        "`podman rm -f %s` on that host to release the resources",
        handle.container_id[:12], handle.host.ssh_dest, handle.container_id[:12],
    )


def stop_container(handle: ContainerHandle) -> None:
    """Forcibly remove the container.

    Safe to call multiple times -- if the container is already gone
    (e.g. ``--rm`` cleanup raced us) we log a warning and return.
    """
    logger.debug("stopping container id=%s", handle.container_id[:12])
    cp = _podman(handle.host, "rm", "-f", handle.container_id, timeout_s=60.0)
    if cp.returncode != 0:
        logger.warning(
            "podman rm -f id=%s failed (already gone?): %s",
            handle.container_id[:12], cp.stderr.decode(errors="replace").strip(),
        )
        return
    logger.info("ephemeral container stopped id=%s", handle.container_id[:12])


def copy_into_container(
    handle: ContainerHandle,
    local_path: Path,
    dest_dir: str,
    *,
    timeout_s: float = 120.0,
) -> None:
    """Copy one local file into the container by streaming a single-file
    tar into ``podman cp -`` over ssh (no host temp file, no quoting).
    ``dest_dir`` is created first."""
    local_path = Path(local_path)
    cp = _podman(handle.host, "exec", handle.container_id, "mkdir", "-p", dest_dir,
                 timeout_s=timeout_s)
    if cp.returncode != 0:
        raise RuntimeError(
            f"mkdir -p {dest_dir} in container failed: "
            f"{cp.stderr.decode(errors='replace').strip()}"
        )
    tar_argv = ["tar", "-C", str(local_path.parent), "-cf", "-", local_path.name]
    cp_argv = handle.host.argv(["podman", "cp", "-", f"{handle.container_id}:{dest_dir}"])
    logger.info("copy into container id=%s file=%s -> %s",
                handle.container_id[:12], local_path, dest_dir)
    tar = subprocess.Popen(tar_argv, stdout=subprocess.PIPE)
    cpp = subprocess.Popen(cp_argv, stdin=tar.stdout,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tar.stdout.close()
    try:
        _, err = cpp.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        cpp.kill()
        cpp.wait()
        tar.kill()
        tar.wait()
        raise
    tar.wait()
    if cpp.returncode != 0:
        raise RuntimeError(
            f"podman cp into container failed: {err.decode(errors='replace').strip()}"
        )


def open_container_shell(
    handle: ContainerHandle,
    agent_remote_path: str,
    *,
    max_output_bytes: int = 256 * 1024,
) -> ContainerShellSession:
    """Open the persistent shell channel into the container:
    ``ssh host podman exec -i <cid> python3 <agent>`` driving the
    ``fairy_agent`` protocol."""
    argv = handle.host.argv([
        "podman", "exec", "-i", handle.container_id, "python3", agent_remote_path,
    ])
    return ContainerShellSession(argv, max_output_bytes=max_output_bytes).start()
