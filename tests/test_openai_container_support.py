import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:  # pragma: no cover - test shim
        pass

    class _FakeError(Exception):
        pass

    fake_openai.OpenAI = _FakeOpenAI
    fake_openai.RateLimitError = _FakeError
    fake_openai.AuthenticationError = _FakeError
    fake_openai.InternalServerError = _FakeError
    sys.modules["openai"] = fake_openai

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.dotenv_values = lambda *_args, **_kwargs: {}
    sys.modules["dotenv"] = fake_dotenv

import openai_container
import openai_container_pool
import podman_repos
import openai_pr_review_wrapper as wrapper
import openai_reviewer


class OpenAIContainerSupportTests(unittest.TestCase):
    def test_normalize_container_repo_cache_defaults(self) -> None:
        cache = openai_container.normalize_container_repo_cache(None)
        self.assertEqual(None, cache["container_id"])
        self.assertEqual(openai_container.DEFAULT_CONTAINER_REPOS_ROOT, cache["container_repos_root"])
        self.assertEqual({}, cache["repo_heads"])
        self.assertEqual({}, cache["repo_entries"])

    def test_save_and_load_container_repo_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = {
                "container_id": "cntr_123",
                "container_repos_root": "/mnt/data/repos",
                "repo_heads": {"/tmp/repo": "abc123"},
                "repo_entries": {
                    "/tmp/repo": {
                        "repo_name": "repo",
                        "head_sha": "abc123",
                        "archive_name": "repo-abc123.tar",
                        "uploaded_file_id": "file_123",
                        "container_file_path": "/mnt/data/repo-abc123.tar",
                        "mounted_path": "/mnt/data/repos/repo",
                    }
                },
            }
            openai_container.save_container_repo_cache(root, cache)

            loaded = openai_container.load_container_repo_cache(root)

        self.assertEqual("cntr_123", loaded["container_id"])
        self.assertEqual("abc123", loaded["repo_heads"]["/tmp/repo"])
        entry = loaded["repo_entries"]["/tmp/repo"]
        self.assertEqual("file_123", entry["uploaded_file_id"])
        self.assertEqual("/mnt/data/repos/repo", entry["mounted_path"])

    def test_build_container_repo_specs_assigns_unique_repo_names(self) -> None:
        repo_roots = [Path("/tmp/work/repo"), Path("/tmp/other/repo")]
        with mock.patch.object(openai_container, "get_repo_head_sha", side_effect=["a" * 40, "b" * 40]):
            specs = openai_container.build_container_repo_specs(
                repo_roots,
                container_repos_root="/mnt/data/repos",
                verbose=False,
            )

        self.assertEqual(["repo", "repo-2"], [spec.repo_name for spec in specs])
        self.assertEqual("/mnt/data/repos/repo", specs[0].mounted_path)
        self.assertEqual("/mnt/data/repos/repo-2", specs[1].mounted_path)
        self.assertTrue(specs[0].archive_name.startswith("repo-aaaaaaaaaaaaaaaa"))
        self.assertTrue(specs[1].archive_name.startswith("repo-2-bbbbbbbbbbbbbbbb"))

    def test_build_container_repo_archive_includes_only_git_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "source"
            (repo_root / ".git").mkdir(parents=True)
            (repo_root / "src").mkdir(parents=True)
            (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (repo_root / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")

            spec = openai_container.ContainerRepoSpec(
                repo_root=repo_root,
                repo_key=str(repo_root),
                repo_name="demo",
                head_sha="a" * 40,
                archive_name="demo.tar",
                mounted_path="/mnt/data/repos/demo",
            )
            archive_path = openai_container.build_container_repo_archive(Path(tmp), spec, verbose=False)

            with tarfile.open(archive_path, "r:") as archive:
                names = set(archive.getnames())

        self.assertIn("demo/.git/HEAD", names)
        self.assertNotIn("demo/src/main.py", names)

    def test_model_visible_repo_head_map_prefers_container_mounts(self) -> None:
        repo_root = Path("/tmp/repo")
        repo_heads = {str(repo_root.resolve()): "a" * 40}
        specs = [
            openai_container.ContainerRepoSpec(
                repo_root=repo_root,
                repo_key=str(repo_root.resolve()),
                repo_name="repo",
                head_sha="a" * 40,
                archive_name="repo-aaaaaaaaaaaaaaaa.tar",
                mounted_path="/mnt/data/repos/repo",
            )
        ]

        visible = wrapper.build_model_visible_repo_head_map(
            [repo_root],
            repo_heads,
            container_repo_specs=specs,
        )

        self.assertEqual({"/mnt/data/repos/repo": "a" * 40}, visible)

    def test_model_visible_repo_head_map_avoids_host_paths_without_container(self) -> None:
        repo_roots = [Path("/tmp/proj/ffmpeg"), Path("/tmp/proj/all_ffmpeg")]
        repo_heads = {
            str(repo_roots[0].resolve()): "a" * 40,
            str(repo_roots[1].resolve()): "b" * 40,
        }

        visible = wrapper.build_model_visible_repo_head_map(
            repo_roots,
            repo_heads,
            container_repo_specs=[],
        )

        self.assertEqual({"ffmpeg": "a" * 40, "all_ffmpeg": "b" * 40}, visible)

    def test_model_visible_repo_head_map_local_specs_use_container_path(self) -> None:
        repo_root = Path("/tmp/ffmpeg")
        specs = [
            podman_repos.RepoSpec(
                repo_root=repo_root,
                name="ffmpeg",
                head_sha="b" * 40,
                container_path="/work/ffmpeg",
                mirror_path="fairy-mirrors/ffmpeg.git",
            )
        ]
        repo_heads = {str(repo_root.resolve()): "b" * 40}
        visible = wrapper.build_model_visible_repo_head_map(
            [repo_root],
            repo_heads,
            podman_repo_specs=specs,
        )
        self.assertEqual({"/work/ffmpeg": "b" * 40}, visible)

    def test_archive_name_change_invalidates_cached_upload_fields(self) -> None:
        spec = openai_container.ContainerRepoSpec(
            repo_root=Path("/tmp/repo"),
            repo_key="/tmp/repo",
            repo_name="repo",
            head_sha="a" * 40,
            archive_name="repo-aaaaaaaaaaaaaaaa.tar",
            mounted_path="/mnt/data/repos/repo",
        )
        existing_entry = {
            "repo_name": "repo",
            "head_sha": "a" * 40,
            "archive_name": "repo-aaaaaaaaaaaaaaaa.tar.gz",
            "uploaded_file_id": "file_old",
            "container_file_path": "/mnt/data/file_old-repo-aaaaaaaaaaaaaaaa.tar.gz",
            "mounted_path": "/mnt/data/repos/repo",
        }
        cache = {"repo_entries": {spec.repo_key: existing_entry}, "repo_heads": {}}
        client = mock.Mock()

        with (
            mock.patch.object(openai_container, "openai_file_exists", return_value=True),
            mock.patch.object(openai_container, "upload_local_file", return_value="file_new"),
            mock.patch.object(openai_container, "build_container_repo_archive", return_value=Path("/tmp/repo.tar")),
        ):
            repo_entries = openai_container.ensure_uploaded_container_repo_archives(
                client,
                [spec],
                cache,
                verbose=False,
            )

        entry = repo_entries[spec.repo_key]
        self.assertEqual("repo-aaaaaaaaaaaaaaaa.tar", entry["archive_name"])
        self.assertEqual("file_new", entry["uploaded_file_id"])
        self.assertNotIn("container_file_path", entry)

    def test_ready_reuses_live_container_from_pool_without_rebootstrap(self) -> None:
        spec = openai_container.ContainerRepoSpec(
            repo_root=Path("/tmp/repo"),
            repo_key="/tmp/repo",
            repo_name="repo",
            head_sha="a" * 40,
            archive_name="repo-aaaaaaaaaaaaaaaa.tar",
            mounted_path="/mnt/data/repos/repo",
        )
        client = mock.Mock()

        with tempfile.TemporaryDirectory() as tmp_pool:
            pool_root = Path(tmp_pool)
            state_hash = openai_container.compute_container_pool_state_hash(
                [spec],
                container_repos_root="/mnt/data/repos",
                memory_limit="1g",
            )
            pool_dir = openai_container_pool.pool_dir_for(pool_root, state_hash)
            pool_dir.mkdir(parents=True, exist_ok=True)
            (pool_dir / "cntr_123").touch()

            with (
                mock.patch.object(openai_container, "build_container_repo_specs", return_value=[spec]),
                mock.patch.object(openai_container, "get_live_container_id", return_value="cntr_123") as live_check,
                mock.patch.object(openai_container, "ensure_uploaded_container_repo_archives", side_effect=AssertionError("unexpected upload")),
                mock.patch.object(openai_container, "create_fresh_openai_container", side_effect=AssertionError("unexpected container creation")),
                mock.patch.object(openai_container, "attach_container_repo_archives", side_effect=AssertionError("unexpected attach")),
                mock.patch.object(openai_container, "upload_container_text_file", side_effect=AssertionError("unexpected script upload")),
                mock.patch.object(openai_container, "bootstrap_openai_container_repos", side_effect=AssertionError("unexpected bootstrap")),
            ):
                lease = openai_container.ensure_openai_container_repos_ready(
                    client,
                    [spec.repo_root],
                    explicit_container_id=None,
                    container_repos_root="/mnt/data/repos",
                    expiry_minutes=20,
                    memory_limit="1g",
                    verbose=False,
                    pool_root=pool_root,
                )

            self.assertEqual("cntr_123", lease.container_id)
            self.assertEqual([spec], lease.specs)
            live_check.assert_called_once_with(client, "cntr_123", verbose=False, strict=False)
            # Pool entry was claimed (unlinked) by acquire.
            self.assertFalse((pool_dir / "cntr_123").exists())

            # Releasing healthy should put it back in the pool.
            self.assertTrue(lease.release(healthy=True))
            self.assertTrue((pool_dir / "cntr_123").exists())

    def test_ready_skips_stale_pool_entries_and_returns_live_one(self) -> None:
        spec = openai_container.ContainerRepoSpec(
            repo_root=Path("/tmp/repo"),
            repo_key="/tmp/repo",
            repo_name="repo",
            head_sha="a" * 40,
            archive_name="repo-aaaaaaaaaaaaaaaa.tar",
            mounted_path="/mnt/data/repos/repo",
        )
        client = mock.Mock()

        with tempfile.TemporaryDirectory() as tmp_pool:
            pool_root = Path(tmp_pool)
            state_hash = openai_container.compute_container_pool_state_hash(
                [spec],
                container_repos_root="/mnt/data/repos",
                memory_limit="1g",
            )
            pool_dir = openai_container_pool.pool_dir_for(pool_root, state_hash)
            pool_dir.mkdir(parents=True, exist_ok=True)
            for name in ("cntr_stale1", "cntr_stale2", "cntr_live"):
                (pool_dir / name).touch()

            def liveness(_client: object, container_id: str, **_kwargs: object) -> str | None:
                return container_id if container_id == "cntr_live" else None

            with (
                mock.patch.object(openai_container, "build_container_repo_specs", return_value=[spec]),
                mock.patch.object(openai_container, "get_live_container_id", side_effect=liveness),
                mock.patch.object(openai_container, "ensure_uploaded_container_repo_archives", side_effect=AssertionError("unexpected upload")),
                mock.patch.object(openai_container, "create_fresh_openai_container", side_effect=AssertionError("unexpected container creation")),
            ):
                lease = openai_container.ensure_openai_container_repos_ready(
                    client,
                    [spec.repo_root],
                    explicit_container_id=None,
                    container_repos_root="/mnt/data/repos",
                    expiry_minutes=20,
                    memory_limit="1g",
                    verbose=False,
                    pool_root=pool_root,
                )

            self.assertEqual("cntr_live", lease.container_id)
            self.assertEqual([], sorted(p.name for p in pool_dir.iterdir()))

    def test_ready_provisions_fresh_when_pool_is_empty(self) -> None:
        spec = openai_container.ContainerRepoSpec(
            repo_root=Path("/tmp/repo"),
            repo_key="/tmp/repo",
            repo_name="repo",
            head_sha="b" * 40,
            archive_name="repo-bbbbbbbbbbbbbbbb.tar",
            mounted_path="/mnt/data/repos/repo",
        )
        client = mock.Mock()
        repo_entries = {
            spec.repo_key: {
                "repo_name": spec.repo_name,
                "head_sha": spec.head_sha,
                "archive_name": spec.archive_name,
                "uploaded_file_id": "file_456",
                "container_file_path": "/mnt/data/file_456-repo-bbbbbbbbbbbbbbbb.tar",
                "mounted_path": spec.mounted_path,
            }
        }

        with tempfile.TemporaryDirectory() as tmp_pool:
            pool_root = Path(tmp_pool)

            with (
                mock.patch.object(openai_container, "build_container_repo_specs", return_value=[spec]),
                mock.patch.object(openai_container, "load_container_repo_cache", return_value={"repo_entries": {}}),
                mock.patch.object(openai_container, "save_container_repo_cache"),
                mock.patch.object(openai_container, "get_live_container_id", side_effect=AssertionError("no liveness check on provision")),
                mock.patch.object(openai_container, "ensure_uploaded_container_repo_archives", return_value=repo_entries) as ensure_uploads,
                mock.patch.object(openai_container, "create_fresh_openai_container", return_value="cntr_456") as create_fresh,
                mock.patch.object(openai_container, "attach_container_repo_archives") as attach_archives,
                mock.patch.object(openai_container, "upload_container_text_file", return_value="/mnt/data/script.py") as upload_script,
                mock.patch.object(openai_container, "bootstrap_openai_container_repos") as bootstrap,
            ):
                lease = openai_container.ensure_openai_container_repos_ready(
                    client,
                    [spec.repo_root],
                    explicit_container_id=None,
                    container_repos_root="/mnt/data/repos",
                    expiry_minutes=20,
                    memory_limit="1g",
                    verbose=False,
                    pool_root=pool_root,
                )

            self.assertEqual("cntr_456", lease.container_id)
            self.assertEqual([spec], lease.specs)
            ensure_uploads.assert_called_once()
            create_fresh.assert_called_once()
            attach_archives.assert_called_once()
            upload_script.assert_called_once()
            bootstrap.assert_called_once()

            # Releasing unhealthy must NOT add the container to the pool.
            self.assertFalse(lease.release(healthy=False))
            state_hash = openai_container.compute_container_pool_state_hash(
                [spec],
                container_repos_root="/mnt/data/repos",
                memory_limit="1g",
            )
            pool_dir = openai_container_pool.pool_dir_for(pool_root, state_hash)
            self.assertFalse((pool_dir / "cntr_456").exists())

    def test_build_response_tools_share_container_between_shell_and_python(self) -> None:
        tools = openai_reviewer.build_response_tools(
            vector_store_ids=["vs_1"],
            file_search_max_num_results=7,
            use_web_search=False,
            web_search_context_size="medium",
            web_search_cache_only=False,
            web_search_domains=[],
            use_shell=True,
            shell_container_id="cntr_123",
            code_interpreter_container_id="cntr_123",
        )

        self.assertEqual("file_search", tools[0]["type"])
        self.assertEqual("shell", tools[1]["type"])
        self.assertEqual("container_reference", tools[1]["environment"]["type"])
        self.assertEqual("cntr_123", tools[1]["environment"]["container_id"])
        self.assertEqual("code_interpreter", tools[2]["type"])
        self.assertEqual("cntr_123", tools[2]["container"])

    def test_build_response_tools_default_code_interpreter_is_auto(self) -> None:
        tools = openai_reviewer.build_response_tools(
            vector_store_ids=[],
            file_search_max_num_results=None,
            use_web_search=False,
            web_search_context_size="medium",
            web_search_cache_only=False,
            web_search_domains=[],
            use_shell=False,
            shell_container_id=None,
            code_interpreter_container_id=None,
        )

        self.assertEqual([{"type": "code_interpreter", "container": {"type": "auto"}}], tools)

    def test_build_response_tools_podman_shell_is_function_only_no_code_interpreter(self) -> None:
        tools = openai_reviewer.build_response_tools(
            vector_store_ids=[],
            file_search_max_num_results=None,
            use_web_search=False,
            web_search_context_size="medium",
            web_search_cache_only=False,
            web_search_domains=[],
            use_shell=False,
            shell_container_id=None,
            code_interpreter_container_id=None,
            use_podman_shell=True,
        )
        self.assertEqual(1, len(tools))
        self.assertEqual("function", tools[0]["type"])
        self.assertEqual("shell", tools[0]["name"])

    def test_build_response_include_omits_code_interpreter_for_podman_shell(self) -> None:
        inc = openai_reviewer.build_response_include(
            vector_store_ids=[],
            use_web_search=False,
            use_podman_shell=True,
        )
        self.assertNotIn("code_interpreter_call.outputs", inc)

    def test_build_container_repo_bootstrap_command_is_single_script_exec(self) -> None:
        command = openai_container.build_container_repo_bootstrap_command("/mnt/data/openai_container_repo_extract.py")

        self.assertEqual('python3 "/mnt/data/openai_container_repo_extract.py"', command)

    def test_build_container_repo_bootstrap_script_uses_expected_paths(self) -> None:
        specs = [
            openai_container.ContainerRepoSpec(
                repo_root=Path("/tmp/repo1"),
                repo_key="/tmp/repo1",
                repo_name="repo1",
                head_sha="abc123",
                archive_name="repo1-abc123.tar",
                mounted_path="/mnt/data/repos/repo1",
            )
        ]
        repo_entries = {
            "/tmp/repo1": {
                "container_file_path": "/mnt/data/repo1-abc123.tar",
            }
        }

        script = openai_container.build_container_repo_bootstrap_script(
            specs,
            repo_entries,
            container_repos_root="/mnt/data/repos",
        )

        self.assertIn("/mnt/data/repo1-abc123.tar", script)
        self.assertIn("/mnt/data/repos/repo1", script)
        self.assertIn(".openai_repo_head", script)
        self.assertIn("tarfile.open", script)

    def test_bootstrap_uses_nano_and_script_path(self) -> None:
        client = mock.Mock()
        client.responses.create.return_value = {"output_text": "READY"}

        with mock.patch.object(openai_container, "extract_response_text", return_value="READY"):
            openai_container.bootstrap_openai_container_repos(
                client,
                container_id="cntr_123",
                script_path="/mnt/data/openai_container_repo_extract.py",
                bootstrap_manifest=[],
                verbose=False,
            )

        kwargs = client.responses.create.call_args.kwargs
        self.assertEqual(openai_container.DEFAULT_CONTAINER_BOOTSTRAP_MODEL, kwargs["model"])
        text = kwargs["input"][0]["content"][0]["text"]
        self.assertIn('python3 "/mnt/data/openai_container_repo_extract.py"', text)

    def test_bootstrap_rejects_not_ready_response(self) -> None:
        client = mock.Mock()
        client.responses.create.return_value = {"output_text": "NOT READY"}

        with mock.patch.object(openai_container, "extract_response_text", return_value="NOT READY"):
            with self.assertRaisesRegex(RuntimeError, "did not confirm readiness"):
                openai_container.bootstrap_openai_container_repos(
                    client,
                    container_id="cntr_123",
                    script_path="/mnt/data/openai_container_repo_extract.py",
                    bootstrap_manifest=[],
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()
