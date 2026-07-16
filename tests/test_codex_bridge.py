"""The MCP stdio bridge between the codex CLI and the shell dispatch socket."""

import json
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codex_bridge
import podman_host
import shell_socket


def _result(out=""):
    return podman_host.ExecResult(
        exit_code=0, stdout=out, stderr="", duration_s=0.01,
        stdout_truncated=False, stderr_truncated=False,
    )


class BridgeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.out: list[dict] = []
        patcher = mock.patch.object(
            codex_bridge, "_write", side_effect=self.out.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _bridge(self, socket_path="/nonexistent", labels=("x86_64", "arm64")):
        return codex_bridge.Bridge(socket_path, list(labels))

    def test_initialize_echoes_client_protocol_version(self) -> None:
        bridge = self._bridge()
        bridge.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2026-01-01"}})
        result = self.out[0]["result"]
        self.assertEqual("2026-01-01", result["protocolVersion"])
        self.assertIn("tools", result["capabilities"])

    def test_tools_list_carries_machine_enum(self) -> None:
        bridge = self._bridge()
        bridge.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        (tool,) = self.out[0]["result"]["tools"]
        self.assertEqual("shell", tool["name"])
        self.assertEqual(
            ["x86_64", "arm64"],
            tool["inputSchema"]["properties"]["machine"]["enum"],
        )

    def test_tools_list_single_machine_has_no_machine_param(self) -> None:
        bridge = self._bridge(labels=("x86_64",))
        bridge.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        (tool,) = self.out[0]["result"]["tools"]
        self.assertNotIn("machine", tool["inputSchema"]["properties"])

    def test_unknown_method_with_id_is_method_not_found(self) -> None:
        bridge = self._bridge()
        bridge.handle({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
        self.assertEqual(
            codex_bridge.METHOD_NOT_FOUND, self.out[0]["error"]["code"])

    def test_notifications_are_ignored(self) -> None:
        bridge = self._bridge()
        bridge.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual([], self.out)

    def test_unknown_tool_is_tool_level_error(self) -> None:
        bridge = self._bridge()
        bridge.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "apply_patch", "arguments": {}}})
        self.assertTrue(self.out[0]["result"]["isError"])

    def test_dispatch_failure_is_jsonrpc_error(self) -> None:
        bridge = self._bridge(socket_path="/nonexistent/shell.sock")
        bridge.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                       "params": {"name": "shell",
                                  "arguments": {"command": "true"}}})
        self.assertEqual(
            codex_bridge.INTERNAL_ERROR, self.out[0]["error"]["code"])

    def test_tools_call_end_to_end_through_socket(self) -> None:
        opened = []

        def open_shell(label):
            opened.append(label)
            session = mock.Mock(spec=podman_host.ContainerShellSession)
            session.exec.return_value = _result(out=f"on {label}\n")
            return session, ""

        server = shell_socket.ShellDispatchServer(
            machine_labels=("x86_64", "arm64"), open_shell=open_shell,
            max_timeout_s=60.0,
        )
        self.addCleanup(server.close)
        bridge = self._bridge(socket_path=server.socket_path)
        for i, machine in enumerate(("x86_64", "arm64")):
            bridge.handle({
                "jsonrpc": "2.0", "id": 10 + i, "method": "tools/call",
                "params": {"name": "shell",
                           "arguments": {"command": "true", "machine": machine}},
            })
            result = self.out[i]["result"]
            self.assertFalse(result["isError"])
            payload = json.loads(result["content"][0]["text"])
            self.assertEqual(f"on {machine}\n", payload["stdout"])
        self.assertEqual(["x86_64", "arm64"], opened)
        self.assertIsNotNone(bridge.client)
        bridge.client.close()


if __name__ == "__main__":
    unittest.main()
