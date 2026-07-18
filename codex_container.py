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

Run a ``codex exec`` pass inside an ephemeral container on a remote podman
host, so codex never executes on the wrapper host (which holds the ssh
key and the provider API keys).

Built on the same podman_host primitives as the review shells --
``start_ephemeral_container`` / ``copy_into_container`` / ``podman exec``
over ``RemoteHost`` ssh -- with no bind mounts: the codex binary is baked
into the image, everything else (auth.json, codex_bridge.py, relay.py,
the prompt and schema) is ``podman cp``'d in per run.

The shell tool reaches the review containers through ``relay.py`` (see
``shell_socket.serve_dispatch``): the wrapper drives ``relay.py`` over a
``podman exec -i`` channel exactly as it drives the review shells, so no
socket ever crosses the host boundary.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Sequence

from podman_host import (
    CONTAINER_CPUS,
    CONTAINER_MEMORY,
    ContainerHandle,
    ContainerShellSession,
    RemoteHost,
    copy_into_container,
    start_ephemeral_container,
    stop_container,
)
from shell_socket import serve_dispatch

logger = logging.getLogger(__name__)

# Under /work = the image WORKDIR.
CONTAINER_RUN_DIR = "/work/.codex-run"
CONTAINER_CODEX_HOME = "/work/.codex-home"
RELAY_SOCKET_PATH = f"{CONTAINER_RUN_DIR}/shell.sock"


class CodexContainer:
    """One ephemeral codex container on a remote podman host.

    Thin lifecycle wrapper over podman_host: ``start`` launches it,
    ``put_file`` / ``put_text`` copy inputs in, ``run`` execs a command
    with the prompt on stdin, ``read_file`` retrieves an output file, and
    ``stop`` removes it. ``--rm`` means a crash cannot leak the container.
    """

    def __init__(
        self,
        *,
        image: str,
        host: RemoteHost,
        memory: str = CONTAINER_MEMORY,
        cpus: str = CONTAINER_CPUS,
        network: str | None = None,
    ) -> None:
        self.image = image
        self.host = host
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.handle: ContainerHandle | None = None

    def start(self) -> "CodexContainer":
        self.handle = start_ephemeral_container(
            image=self.image, host=self.host, network=self.network,
            memory=self.memory, cpus=self.cpus,
        )
        return self

    def _cid(self) -> str:
        if self.handle is None:
            raise RuntimeError("codex container not started")
        return self.handle.container_id

    def exec_argv(
        self,
        remote_argv: Sequence[str],
        *,
        interactive: bool = False,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """``ssh host podman exec [-i] [-e K=V...] cid <remote_argv>``."""
        flags = ["exec"]
        if interactive:
            flags.append("-i")
        for key, value in (env or {}).items():
            flags += ["-e", f"{key}={value}"]
        return self.host.argv(["podman", *flags, self._cid(), *remote_argv])

    def put_file(self, local_path: Path, dest_dir: str,
                 *, timeout_s: float = 120.0) -> None:
        copy_into_container(self.handle, Path(local_path), dest_dir,
                            timeout_s=timeout_s)

    def put_text(self, text: str, dest_dir: str, filename: str,
                 *, timeout_s: float = 120.0) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / filename
            local.write_text(text, encoding="utf-8")
            self.put_file(local, dest_dir, timeout_s=timeout_s)

    def read_file(self, container_path: str,
                  *, timeout_s: float = 60.0) -> str | None:
        """``cat`` a file out of the container; ``None`` if it is absent."""
        cp = subprocess.run(
            self.exec_argv(["cat", container_path]),
            capture_output=True, text=True, timeout=timeout_s,
        )
        return cp.stdout if cp.returncode == 0 else None

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess:
        """Exec ``argv`` in the container with ``input_text`` on stdin."""
        return subprocess.run(
            self.exec_argv(argv, interactive=True, env=env),
            input=input_text, capture_output=True, text=True,
            timeout=timeout_s,
        )

    def stop(self) -> None:
        if self.handle is not None:
            stop_container(self.handle)
            self.handle = None


class CodexShellRelay:
    """Drive ``relay.py`` in the codex container and service its shell calls.

    ``start`` execs ``relay.py`` over ``podman exec -i`` and, once it
    reports its container-local socket is bound, pumps every shell-tool
    request the bridge sends up that channel to the review containers via
    ``serve_dispatch`` -- the container side of the same JSON-over-ssh
    transport the review shells use. ``stop`` tears the channel down.
    """

    def __init__(
        self,
        container: CodexContainer,
        *,
        relay_container_path: str,
        machine_labels: Sequence[str],
        open_shell: Callable[[str], tuple[ContainerShellSession, str]],
        max_timeout_s: float,
        socket_path: str = RELAY_SOCKET_PATH,
    ) -> None:
        self.container = container
        self.relay_container_path = relay_container_path
        self.socket_path = socket_path
        self.machine_labels = tuple(machine_labels)
        self.open_shell = open_shell
        self.max_timeout_s = max_timeout_s
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self, *, ready_timeout_s: float = 30.0) -> "CodexShellRelay":
        argv = self.container.exec_argv(
            ["python3", self.relay_container_path, self.socket_path],
            interactive=True,
        )
        logger.info("codex relay start socket=%s", self.socket_path)
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # relay.py prints RELAY-READY once its socket is bound; wait for it
        # so codex_bridge can never race an unbound socket. A dead relay
        # closes stderr (empty read) -> we raise rather than hang codex.
        ready = self._read_ready(ready_timeout_s)
        if b"RELAY-READY" not in ready:
            self.stop()
            raise RuntimeError(f"codex relay failed to start: {ready!r}")
        self._thread = threading.Thread(
            target=serve_dispatch,
            args=(self._proc.stdout, self._proc.stdin),
            kwargs=dict(machine_labels=self.machine_labels,
                        open_shell=self.open_shell,
                        max_timeout_s=self.max_timeout_s),
            name="codex-shell-dispatch", daemon=True,
        )
        self._thread.start()
        return self

    def _read_ready(self, timeout_s: float) -> bytes:
        result: dict[str, bytes] = {}

        def _read() -> None:
            result["line"] = self._proc.stderr.readline()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout_s)
        return result.get("line", b"")

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.kill()
        self._proc.wait()
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                stream.close()
            except OSError:
                pass
        self._proc = None
