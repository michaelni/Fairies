"""Unit tests for the vendor-agnostic ``podman_host`` module.

Podman is always reached as ``ssh host podman ...``; these tests mock
subprocess so no real podman/ssh is needed, locking down argv
composition (the ssh wrapping, tar-streamed build context, podman cp
tar stream) and the persistent ``ContainerShellSession`` protocol. The
session tests run the *real* in-container agent
(``containers/fairy_agent.py``) directly over a pipe -- no container -- so
they double as the regression test for the live transport.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import podman_host as lc  # noqa: E402

AGENT = REPO_ROOT / "containers" / "fairy_agent.py"
HOST = lc.RemoteHost("fairy@h")
SSH_PREFIX = [
    "ssh", "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
    "-o", "LogLevel=ERROR", "fairy@h",
]


def _ssh(remote_cmd: str) -> list[str]:
    """Expected argv for a fixed-shape podman command run over HOST."""
    return [*SSH_PREFIX, remote_cmd]


def _completed(returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> mock.Mock:
    cp = mock.Mock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


class _FakePopen:
    """subprocess.Popen stand-in for the tar|build and tar|cp pipelines."""

    def __init__(self, argv, rc: int = 0, *, comm=(b"", b""), **_kwargs) -> None:
        self.argv = argv
        self._rc = rc
        self.returncode = rc
        self.stdout = mock.Mock()
        self._comm = comm

    def wait(self, timeout=None):  # noqa: ARG002
        return self._rc

    def communicate(self, timeout=None):  # noqa: ARG002
        self.returncode = self._rc
        return self._comm

    def kill(self) -> None:
        pass


class RemoteHostTests(unittest.TestCase):
    def test_argv_shlex_joins_remote_command(self) -> None:
        argv = HOST.argv(["podman", "cp", "/m/a b/.git", "cid:/work/a/.git"])
        self.assertEqual(_ssh("podman cp '/m/a b/.git' cid:/work/a/.git"), argv)

    def test_argv_adds_identity_when_set(self) -> None:
        host = lc.RemoteHost("fairy@h", identity="/k/id")
        self.assertEqual(
            ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
             "-o", "ServerAliveCountMax=3", "-o", "LogLevel=ERROR",
             "-i", "/k/id", "fairy@h", "true"],
            host.argv(["true"]),
        )

    def test_run_on_remote_host_captures_result(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0, stdout=b"ok")
            res = lc.run_on_remote_host(HOST, "podman", "cp", "/m/.git", "cid:/work/.git")
        self.assertEqual(0, res.returncode)
        self.assertEqual(b"ok", res.stdout)
        self.assertEqual(_ssh("podman cp /m/.git cid:/work/.git"), run.call_args.args[0])


class ImageTagExistsTests(unittest.TestCase):
    def test_true_when_podman_returns_zero(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0)
            self.assertTrue(lc.image_tag_exists("localhost/x:y", host=HOST))
        self.assertEqual(_ssh("podman image exists localhost/x:y"), run.call_args.args[0])

    def test_false_when_missing(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(1)
            self.assertFalse(lc.image_tag_exists("nope:tag", host=HOST))


class EnsureIsolatedNetworkTests(unittest.TestCase):
    def test_skips_create_when_network_exists(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0)
            lc.ensure_isolated_network("fairy-isolated", host=HOST)
        self.assertEqual(1, run.call_count)
        self.assertEqual(_ssh("podman network exists fairy-isolated"),
                         run.call_args_list[0].args[0])

    def test_creates_network_when_missing(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.side_effect = [_completed(1), _completed(0)]
            lc.ensure_isolated_network("fairy-isolated", host=HOST)
        self.assertEqual(2, run.call_count)
        self.assertEqual(_ssh("podman network create fairy-isolated"),
                         run.call_args_list[1].args[0])

    def test_raises_on_create_failure(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.side_effect = [_completed(1), _completed(125, stderr=b"boom")]
            with self.assertRaisesRegex(RuntimeError, "boom"):
                lc.ensure_isolated_network("fairy-isolated", host=HOST)


class BuildImageIfNeededTests(unittest.TestCase):
    def _build(self, *, run_rc, popen_rcs, **kwargs):
        """Run build_image_if_needed with subprocess.run (image-exists
        precheck) and subprocess.Popen (tar | build) mocked. Returns the
        run mock and the list of Popen argvs."""
        popen_argvs: list = []

        def fake_popen(argv, **_kw):
            popen_argvs.append(argv)
            return _FakePopen(argv, rc=popen_rcs[len(popen_argvs) - 1])

        with mock.patch.object(lc.subprocess, "run", return_value=_completed(run_rc)) as run, \
                mock.patch.object(lc.subprocess, "Popen", side_effect=fake_popen):
            lc.build_image_if_needed(host=HOST, **kwargs)
        return run, popen_argvs

    def test_skips_build_when_image_exists(self) -> None:
        with mock.patch.object(lc.subprocess, "run", return_value=_completed(0)) as run, \
                mock.patch.object(lc.subprocess, "Popen") as popen:
            lc.build_image_if_needed(
                image_tag="fairy:latest",
                dockerfile=Path("/c/Containerfile"), context_dir=Path("/c"), host=HOST,
            )
        self.assertEqual(_ssh("podman image exists fairy:latest"), run.call_args.args[0])
        popen.assert_not_called()

    def test_builds_streaming_tar_context_over_ssh(self) -> None:
        run, popen_argvs = self._build(
            run_rc=1, popen_rcs=[0, 0],
            image_tag="fairy:latest",
            dockerfile=Path("/tmp/ctx/Containerfile"), context_dir=Path("/tmp/ctx"),
        )
        self.assertEqual(_ssh("podman image exists fairy:latest"), run.call_args.args[0])
        self.assertEqual(["tar", "-C", "/tmp/ctx", "-cf", "-", "."], popen_argvs[0])
        self.assertEqual(_ssh("podman build -t fairy:latest -f Containerfile -"),
                         popen_argvs[1])

    def test_build_inherits_stdio_no_capture(self) -> None:
        # Regression: build output (apt-get progress) must stream live, not
        # be captured and dumped only on failure.
        popen_kwargs: list = []

        def fake_popen(argv, **kw):
            popen_kwargs.append(kw)
            return _FakePopen(argv, rc=0)

        with mock.patch.object(lc.subprocess, "run", return_value=_completed(1)), \
                mock.patch.object(lc.subprocess, "Popen", side_effect=fake_popen):
            lc.build_image_if_needed(
                image_tag="fairy:latest",
                dockerfile=Path("/c/Containerfile"), context_dir=Path("/c"), host=HOST,
            )
        build_kwargs = popen_kwargs[1]
        self.assertNotIn("stdout", build_kwargs)
        self.assertNotIn("stderr", build_kwargs)
        self.assertNotIn("capture_output", build_kwargs)

    def test_force_rebuilds_without_precheck(self) -> None:
        run, popen_argvs = self._build(
            run_rc=0, popen_rcs=[0, 0],
            image_tag="fairy:latest",
            dockerfile=Path("/c/Containerfile"), context_dir=Path("/c"), force=True,
        )
        run.assert_not_called()
        self.assertEqual(2, len(popen_argvs))

    def test_build_failure_surfaces_rc(self) -> None:
        with mock.patch.object(lc.subprocess, "run", return_value=_completed(1)), \
                mock.patch.object(lc.subprocess, "Popen",
                                  side_effect=lambda argv, **kw: _FakePopen(argv, rc=2)):
            with self.assertRaisesRegex(RuntimeError, r"rc=2"):
                lc.build_image_if_needed(
                    image_tag="fairy:latest",
                    dockerfile=Path("/c/Containerfile"), context_dir=Path("/c"), host=HOST,
                )

    def test_build_injects_labels(self) -> None:
        _, popen_argvs = self._build(
            run_rc=1, popen_rcs=[0, 0],
            image_tag="fairy:latest",
            dockerfile=Path("/tmp/ctx/Containerfile"), context_dir=Path("/tmp/ctx"),
            labels={"fairy.containerfile-sha256": "deadbeef"},
        )
        self.assertEqual(
            _ssh("podman build -t fairy:latest "
                 "--label fairy.containerfile-sha256=deadbeef -f Containerfile -"),
            popen_argvs[1],
        )

    def test_dockerfile_outside_context_raises(self) -> None:
        with mock.patch.object(lc.subprocess, "run", return_value=_completed(1)), \
                mock.patch.object(lc.subprocess, "Popen"):
            with self.assertRaisesRegex(RuntimeError, "must live inside build context"):
                lc.build_image_if_needed(
                    image_tag="fairy:latest",
                    dockerfile=Path("/elsewhere/Containerfile"),
                    context_dir=Path("/tmp/ctx"), host=HOST,
                )


class ReapStaleContainersTests(unittest.TestCase):
    def _reap(self, run_side_effect, **kw):
        with mock.patch.object(lc.subprocess, "run",
                               side_effect=run_side_effect) as run:
            n = lc.reap_stale_containers(
                HOST, images=["localhost/fairy-review:latest"],
                older_than="60m", **kw)
        return n, run

    def test_lists_stopped_old_containers_then_removes(self) -> None:
        calls = []

        def side_effect(argv, **kw):
            calls.append(argv[-1])
            if "podman ps" in argv[-1]:
                return _completed(0, stdout=b"aaa\nbbb\n")
            return _completed(0)

        n, _ = self._reap(side_effect)
        self.assertEqual(2, n)
        ps = calls[0]
        self.assertIn("ancestor=localhost/fairy-review:latest", ps)
        self.assertIn("until=60m", ps)
        self.assertIn("status=created", ps)
        self.assertIn("status=exited", ps)
        self.assertNotIn("status=running", ps)
        self.assertNotIn("status=paused", ps)
        self.assertIn("podman rm -f aaa bbb", calls[1])

    def test_no_matches_skips_rm(self) -> None:
        def side_effect(argv, **kw):
            return _completed(0, stdout=b"")

        n, run = self._reap(side_effect)
        self.assertEqual(0, n)
        self.assertEqual(1, run.call_count)  # ps only, no rm

    def test_list_failure_is_swallowed(self) -> None:
        def side_effect(argv, **kw):
            return _completed(1, stderr=b"boom")

        n, _ = self._reap(side_effect)
        self.assertEqual(0, n)  # best-effort, no raise


class ImageLabelTests(unittest.TestCase):
    def test_returns_label_value(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0, stdout=b"deadbeef\n")
            value = lc.image_label("fairy:latest", "fairy.containerfile-sha256", host=HOST)
        self.assertEqual("deadbeef", value)
        self.assertEqual(
            _ssh('podman image inspect --format '
                 "'{{ index .Config.Labels \"fairy.containerfile-sha256\" }}' fairy:latest"),
            run.call_args.args[0],
        )

    def test_none_when_image_absent(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(1, stderr=b"no such image")
            self.assertIsNone(lc.image_label("nope:tag", "k", host=HOST))

    def test_none_when_label_empty(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0, stdout=b"\n")
            self.assertIsNone(lc.image_label("fairy:latest", "k", host=HOST))


class StartStopContainerTests(unittest.TestCase):
    def test_start_passes_resource_limits_and_returns_handle(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0, stdout=b"deadbeef0123\n")
            handle = lc.start_ephemeral_container(
                image="fairy:latest", host=HOST,
                network="fairy-isolated", memory="2g", cpus="1",
            )
        argv = run.call_args.args[0]
        self.assertEqual(SSH_PREFIX, argv[:len(SSH_PREFIX)])
        remote = argv[-1]
        for token in ("podman run -d --rm", "--network=fairy-isolated",
                      "--memory=2g", "--cpus=1", "fairy:latest sleep infinity"):
            self.assertIn(token, remote)
        # No `-w`: the image's WORKDIR sets the cwd; podman 4.9.3 rejects
        # `run --workdir` on the WORKDIR-created dir (see start_ephemeral_container).
        self.assertNotIn("-w", remote.split())
        self.assertEqual("deadbeef0123", handle.container_id)
        self.assertEqual("fairy:latest", handle.image)
        self.assertEqual("fairy-isolated", handle.network)
        self.assertEqual(HOST, handle.host)

    def test_start_omits_network_when_unset(self) -> None:
        for network in (None, ""):
            with mock.patch.object(lc.subprocess, "run") as run:
                run.return_value = _completed(0, stdout=b"deadbeef0123\n")
                handle = lc.start_ephemeral_container(
                    image="fairy:latest", host=HOST, network=network,
                )
            self.assertNotIn("--network", run.call_args.args[0][-1])
            self.assertEqual(network, handle.network)

    def test_start_raises_on_empty_id(self) -> None:
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(0, stdout=b"   \n")
            with self.assertRaisesRegex(RuntimeError, "empty container id"):
                lc.start_ephemeral_container(image="x", host=HOST, network="n")

    def test_stop_calls_rm_force_and_swallows_already_gone(self) -> None:
        handle = lc.ContainerHandle(container_id="cidcidcid", image="x",
                                    network="n", host=HOST)
        with mock.patch.object(lc.subprocess, "run") as run:
            run.return_value = _completed(1, stderr=b"no such container")
            lc.stop_container(handle)
        self.assertEqual(_ssh("podman rm -f cidcidcid"), run.call_args.args[0])


class CopyAndOpenShellTests(unittest.TestCase):
    def test_copy_into_container_mkdirs_then_streams_tar(self) -> None:
        handle = lc.ContainerHandle(container_id="cid", image="i", network=None, host=HOST)
        popen_argvs: list = []

        def fake_popen(argv, **_kw):
            popen_argvs.append(argv)
            return _FakePopen(argv, rc=0)

        with mock.patch.object(lc.subprocess, "run", return_value=_completed(0)) as run, \
                mock.patch.object(lc.subprocess, "Popen", side_effect=fake_popen):
            lc.copy_into_container(handle, Path("/x/agent.py"), "/work/.fairy")
        self.assertEqual(_ssh("podman exec cid mkdir -p /work/.fairy"),
                         run.call_args.args[0])
        self.assertEqual(["tar", "-C", "/x", "-cf", "-", "agent.py"], popen_argvs[0])
        self.assertEqual(_ssh("podman cp - cid:/work/.fairy"), popen_argvs[1])

    def test_copy_raises_when_cp_fails(self) -> None:
        handle = lc.ContainerHandle(container_id="cid", image="i", network=None, host=HOST)

        def fake_popen(argv, **_kw):
            return _FakePopen(argv, rc=1, comm=(b"", b"cp boom"))

        with mock.patch.object(lc.subprocess, "run", return_value=_completed(0)), \
                mock.patch.object(lc.subprocess, "Popen", side_effect=fake_popen):
            with self.assertRaisesRegex(RuntimeError, "cp boom"):
                lc.copy_into_container(handle, Path("/x/agent.py"), "/work/.fairy")

    def test_open_container_shell_builds_ssh_exec_argv(self) -> None:
        handle = lc.ContainerHandle(container_id="cid", image="i", network=None, host=HOST)
        captured: dict = {}

        class _FakeSession:
            def __init__(self, argv, **kw):
                captured["argv"] = argv
                captured["kw"] = kw

            def start(self):
                return self

        with mock.patch.object(lc, "ContainerShellSession", _FakeSession):
            lc.open_container_shell(handle, "/work/.fairy/fairy_agent.py",
                                    max_output_bytes=4096)
        self.assertEqual(_ssh("podman exec -i cid python3 /work/.fairy/fairy_agent.py"),
                         captured["argv"])
        self.assertEqual(4096, captured["kw"]["max_output_bytes"])


class _SessionHarness:
    """Shared driver: a real ContainerShellSession over a pipe subprocess."""

    def _session(self, argv, **kw) -> lc.ContainerShellSession:
        s = lc.ContainerShellSession(argv, **kw).start()
        self.addCleanup(s.close)
        return s

    def _agent(self, **kw) -> lc.ContainerShellSession:
        return self._session([sys.executable, str(AGENT)], **kw)


class ContainerShellSessionTests(_SessionHarness, unittest.TestCase):
    """Drive the real fairy_agent over a pipe (argv = python3 agent)."""

    def test_exec_returns_execresult(self) -> None:
        r = self._agent().exec("echo hi")
        self.assertEqual(0, r.exit_code)
        self.assertEqual("hi\n", r.stdout)
        self.assertFalse(r.stdout_truncated)

    def test_one_process_serves_many_execs(self) -> None:
        s = self._agent()
        self.assertEqual("a\n", s.exec("echo a").stdout)
        self.assertEqual(7, s.exec("exit 7").exit_code)
        self.assertEqual("b\n", s.exec("echo b").stdout)

    def test_cwd_and_stderr(self) -> None:
        r = self._agent().exec("pwd; echo e >&2", cwd="/tmp")
        self.assertEqual("/tmp\n", r.stdout)
        self.assertEqual("e\n", r.stderr)

    def test_truncation_flag(self) -> None:
        s = self._agent(max_output_bytes=512)
        r = s.exec("yes ABCDEFGH | head -c 50000")
        self.assertTrue(r.stdout_truncated)
        self.assertLessEqual(len(r.stdout), 512)

    def test_command_timeout_is_124_and_channel_survives(self) -> None:
        s = self._agent()
        self.assertEqual(124, s.exec("sleep 30", timeout_s=0.5).exit_code)
        self.assertEqual("ok\n", s.exec("echo ok").stdout)

    def test_channel_eof_raises(self) -> None:
        s = self._session([sys.executable, "-c", "import sys; sys.stdin.readline()"])
        with self.assertRaises(RuntimeError):
            s.exec("echo hi")

    def test_unresponsive_agent_times_out(self) -> None:
        s = self._session(
            [sys.executable, "-c", "import sys,time; sys.stdin.readline(); time.sleep(30)"]
        )
        with mock.patch.object(lc, "HOST_RESPONSE_MARGIN_S", 0.3):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                s.exec("echo hi", timeout_s=0.1)


class HostileFrameTests(_SessionHarness, unittest.TestCase):
    """Frames a compromised container can inject into the agent's stdout
    (PR-derived code shares the container and can open the agent's
    ``/proc/<pid>/fd/1``). The host must reject each with a clean
    RuntimeError and a dead channel -- never crash with a raw parse error
    or interpret the forged frame."""

    def _evil(self, payload: bytes, **kw) -> lc.ContainerShellSession:
        code = ("import sys\n"
                "sys.stdin.readline()\n"
                f"sys.stdout.buffer.write({payload!r})\n"
                "sys.stdout.buffer.flush()\n"
                "sys.stdin.read()\n")
        return self._session([sys.executable, "-c", code], **kw)

    def _assert_rejected(self, payload: bytes, **kw) -> None:
        s = self._evil(payload, **kw)
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            s.exec("echo hi")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            s.exec("echo again")

    def test_not_json(self) -> None:
        self._assert_rejected(b"segfault: core dumped\n")

    def test_invalid_utf8(self) -> None:
        self._assert_rejected(b'\xff\xfe{"id": 0}\n')

    def test_non_object_frames(self) -> None:
        for payload in (b"[1, 2, 3]\n", b'"a string"\n', b"null\n", b"7\n"):
            with self.subTest(payload=payload):
                self._assert_rejected(payload)

    def test_forged_id_desyncs(self) -> None:
        self._assert_rejected(b'{"id": 7, "exit_code": 0}\n')

    def test_missing_fields(self) -> None:
        self._assert_rejected(b'{"id": 0}\n')

    def test_wrong_typed_exit_code(self) -> None:
        self._assert_rejected(
            b'{"id": 0, "exit_code": {"a": 1}, "stdout_b64": "", '
            b'"stderr_b64": "", "stdout_truncated": false, '
            b'"stderr_truncated": false}\n')

    def test_infinite_exit_code(self) -> None:
        # 1e999 parses as float inf; int(inf) is OverflowError, which must
        # not escape raw.
        self._assert_rejected(
            b'{"id": 0, "exit_code": 1e999, "stdout_b64": "", '
            b'"stderr_b64": "", "stdout_truncated": false, '
            b'"stderr_truncated": false}\n')

    def test_huge_int_literal(self) -> None:
        # Overflows the int digit limit inside json.loads itself.
        self._assert_rejected(
            b'{"id": 0, "exit_code": ' + b"9" * 5000 + b'}\n')

    def test_bad_base64(self) -> None:
        self._assert_rejected(
            b'{"id": 0, "exit_code": 0, "stdout_b64": "A", '
            b'"stderr_b64": "", "stdout_truncated": false, '
            b'"stderr_truncated": false}\n')

    def test_oversized_frame(self) -> None:
        s = self._evil(b"A" * 70_000, max_output_bytes=1024)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            s.exec("echo hi")

    def test_nonfinite_duration_is_replaced(self) -> None:
        s = self._evil(
            b'{"id": 0, "exit_code": 0, "stdout_b64": "aGk=", '
            b'"stderr_b64": "", "stdout_truncated": false, '
            b'"stderr_truncated": false, "duration_s": NaN}\n')
        r = s.exec("echo hi")
        self.assertEqual("hi", r.stdout)
        self.assertTrue(math.isfinite(r.duration_s))


if __name__ == "__main__":
    unittest.main()
