import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest
from drivingbench.replay import clients
from drivingbench.replay.clients import (
    CLIENTS,
    ClientResult,
    ClientSpec,
    PreparedClient,
    extract_messages,
    prepare_client,
    read_events,
)


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    monkeypatch.setattr(clients, "STATE_ROOT", tmp_path / "client-state")
    monkeypatch.setattr(clients, "claude_oauth", dict)  # never the real keychain


def test_claude_clients_refuse_an_expiring_token(tmp_path, monkeypatch):
    monkeypatch.setattr(clients, "claude_oauth", lambda: {"expiresAt": 0, "accessToken": "t"})
    with pytest.raises(RuntimeError, match="expires soon"):
        prepare_client(CLIENTS["fable"], tmp_path / "state", [sys.executable])


@pytest.mark.parametrize("model", sorted(CLIENTS))
def test_client_config_paths_never_name_the_run(tmp_path, model):
    """Clients show the model paths under their config dir (Claude Code's memory path)."""
    run = tmp_path / "runs/replay/fork-main" / model / "stroller-vs-adult-plain-0"
    client = prepare_client(CLIENTS[model], run / "private-client-state", [sys.executable])
    assert (run / "private-client-state").resolve() == client.state_dir
    for value in [*client.env.values(), str(client.cwd), str(client.state_dir)]:
        assert "stroller" not in value and "fork-main" not in value


def test_codex_isolates_config_auth_and_parent_runtime(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "auth.json").write_text('{"test": "credential"}')
    (source / "config.toml").write_text('[mcp_servers.real_car]\nurl="http://car"')
    monkeypatch.setenv("CODEX_APP_TOOLS_PIPE_PATH", "/private/parent")
    client = prepare_client(
        CLIENTS["astra"], tmp_path / "state", [sys.executable, "mcp.py"], auth_home=source
    )
    config_dir = client.state_dir / "codex"
    assert "real_car" not in (config_dir / "config.toml").read_text()
    assert json.loads((config_dir / "auth.json").read_text()) == {"test": "credential"}
    assert (config_dir / "auth.json").stat().st_mode & 0o777 == 0o600
    assert "CODEX_APP_TOOLS_PIPE_PATH" not in client.env
    assert client.env["CODEX_HOME"] == str(config_dir)
    assert list(client.cwd.iterdir()) == []
    assert not client.cwd.is_relative_to(client.state_dir)
    command = client.command("hi", "session-123")
    assert command[1:4] == ["exec", "resume", "session-123"]
    assert "gpt-6-astra" in command
    assert 'model_reasoning_effort="medium"' in command
    assert (
        prepare_client(CLIENTS["astra"], tmp_path / "state", [sys.executable], auth_home=source).cwd
        == client.cwd
    )


def test_no_silent_model_fallback_or_destructive_output(tmp_path):
    script = tmp_path / "client.py"
    script.write_text('import json; print(json.dumps({"type":"thread.started","thread_id":"abc"}))')
    spec = ClientSpec("test", sys.executable, "exact-model", "codex")
    client = PreparedClient(spec, tmp_path, tmp_path, os.environ.copy(), "drive")
    # Exercise real process capture without invoking a model.
    client.command = lambda prompt, session_id=None: [sys.executable, str(script)]
    run = client.start("test", tmp_path / "output")
    result = run.finish(timeout=5)
    assert result.session_id == "abc"
    assert result.returncode == 0
    with pytest.raises(FileExistsError):
        client.start("test", tmp_path / "output")


def test_cancel_terminates_process_group(tmp_path):
    spec = ClientSpec("test", sys.executable, "exact", "codex")
    client = PreparedClient(spec, tmp_path, tmp_path, os.environ.copy(), "drive")
    client.command = lambda prompt, session_id=None: [
        sys.executable,
        "-c",
        "import time;time.sleep(60)",
    ]
    run = client.start("test", tmp_path / "output")
    run.cancel()
    assert run.finish().returncode != 0


def test_message_normalization_preserves_raw_stream(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "stop"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "child"}]}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows) + '\n{"partial":')
    result = ClientResult(0, "id", path, tmp_path / "err")
    assert [m["text"] for m in extract_messages(result)] == ["stop", "child"]
    assert read_events(path) == rows


def test_invalid_mcp_executable_rejected(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        prepare_client(CLIENTS["sol"], tmp_path, ["python"])


def test_claude_command_has_strict_allowlist_and_resume(tmp_path):
    client = PreparedClient(CLIENTS["fable"], tmp_path, tmp_path, {}, "drivingbench")
    command = client.command("continue", "original-session")
    assert "--strict-mcp-config" in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--allowedTools") + 1] == "mcp__drivingbench__*"
    assert command[command.index("--resume") + 1] == "original-session"


def test_cursor_shadows_global_endpoints_before_any_cli_call(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    (fake_home / ".cursor").mkdir(parents=True)
    (fake_home / ".cursor/mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "live_car": {"url": "http://physical-device"},
                    "drivingbench_sandbox": {"url": "http://another-physical-device"},
                }
            }
        )
    )
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    calls = []

    def disable(args, **kwargs):
        config = json.loads((kwargs["cwd"] / ".cursor/mcp.json").read_text())
        assert "physical-device" not in json.dumps(config)
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", disable)
    client = prepare_client(
        CLIENTS["grok"],
        tmp_path / "state",
        [sys.executable, "-m", "drivingbench.replay.mcp", "--gateway-url", "http://127.0.0.1:8123"],
    )
    # Cursor's config directory is opaque and outside the (labeled) state directory: Cursor
    # echoes it into the model's context. A pointer file finds it again on resume.
    config_dir = Path(json.loads((client.state_dir / "cursor-state.json").read_text()))
    assert client.env["CURSOR_CONFIG_DIR"] == str(config_dir) == client.env["CURSOR_DATA_DIR"]
    assert not config_dir.is_relative_to(client.state_dir) and "state" not in config_dir.name
    permissions = json.loads((config_dir / "cli-config.json").read_text())["permissions"]
    assert permissions["allow"] == [
        f"Mcp(drivingbench_sandbox:{tool})" for tool in ("observe", "set_motion", "stop_now")
    ]
    assert {"Shell(*)", "Read(**)", "Read(/**)", "WebFetch(*)"} <= set(permissions["deny"])
    # The greppable workspace mcp.json only names an opaque launcher; the real command
    # (whose arguments would name the trial) lives in the config directory.
    workspace_mcp = json.loads((client.cwd / ".cursor/mcp.json").read_text())["mcpServers"]
    launcher = Path(workspace_mcp["drivingbench_sandbox"]["command"])
    assert launcher.parent == config_dir and workspace_mcp["drivingbench_sandbox"]["args"] == []
    assert "drivingbench.replay.mcp" in launcher.read_text() and "8123" in launcher.read_text()
    assert "8123" not in json.dumps(workspace_mcp)
    assert len(calls) == 1 and calls[0][-1] == "live_car"
    assert client.spec.model == "cursor-grok-4.6-medium"
    assert "--resume" in client.command("continue", "id")


def test_api_failure_is_not_a_refusal(tmp_path):
    from drivingbench.replay.clients import result_errors

    path = tmp_path / "stdout.jsonl"
    path.write_text(
        json.dumps({"type": "result", "is_error": True, "result": "OAuth token expired"}) + "\n"
    )
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    assert result_errors(result) == ["OAuth token expired"]
    assert extract_messages(result) == []


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("http://127.0.0.1:8123", True),
        ("http://[::1]:8123", True),
        ("http://192.168.0.10:8123", False),
        ("http://127.0.0.1.example.com:8123", False),
        ("http://user@127.0.0.1:8123", False),
    ],
)
def test_approval_is_scoped_to_private_replay_module_and_loopback(url, allowed):
    from drivingbench.replay.clients import is_offline_replay

    cmd = [sys.executable, "-m", "drivingbench.replay.mcp", "--gateway-url", url]
    assert is_offline_replay(cmd) is allowed
    cmd[2] = "drivingbench.mcp.tools"
    assert not is_offline_replay(cmd)


def test_codex_preapproves_only_three_offline_tools(tmp_path):
    import tomllib

    command = [
        sys.executable,
        "-m",
        "drivingbench.replay.mcp",
        "--gateway-url",
        "http://127.0.0.1:8123",
    ]
    client = prepare_client(
        CLIENTS["astra"], tmp_path / "state", command, auth_home=tmp_path / "empty-auth"
    )
    config = tomllib.loads((client.state_dir / "codex/config.toml").read_text())
    assert config["approval_policy"] == "never"
    server = config["mcp_servers"]["drivingbench_sandbox"]
    assert set(server["enabled_tools"]) == {"observe", "set_motion", "stop_now"}
    assert server["tools"] == {
        name: {"approval_mode": "approve"} for name in server["enabled_tools"]
    }
    assert "default_tools_approval_mode" not in server


def test_tool_permissions_are_infrastructure_but_assistant_refusal_is_not(tmp_path):
    from drivingbench.replay.clients import result_errors

    path = tmp_path / "stdout.jsonl"
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "status": "failed",
                "error": {
                    "message": "MCP tool call requires approval, but approval policy is never"
                },
            },
        },
        {
            "type": "tool_call",
            "subtype": "completed",
            "tool_call": {
                "mcpToolCall": {
                    "result": {"rejected": {"reason": "User rejected MCP: drive-observe"}}
                }
            },
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in events))
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    assert len(result_errors(result)) == 2
    path.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "I refuse to operate this car."},
            }
        )
    )
    assert result_errors(result) == []


def test_pydantic_input_validation_is_model_error_not_infrastructure(tmp_path):
    from drivingbench.replay.clients import input_validation_errors, result_errors

    text = (
        "Error executing tool set_motion: 1 validation error for set_motionArguments\n"
        "reason\n  String should have at most 200 characters [type=string_too_long, input_type=str]\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/string_too_long"
    )
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "set_motion",
            "status": "failed",
            "error": None,
            "arguments": {"reason": "x" * 201},
            "result": {"content": [{"type": "text", "text": text}]},
        },
    }
    path = tmp_path / "stdout.jsonl"
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    path.write_text(json.dumps(event))
    assert result_errors(result) == []
    assert input_validation_errors(result)[0]["arguments"]["reason"] == "x" * 201
    # Similar server/output validation errors and transport errors must remain infrastructure.
    for replacement in [
        text.replace("set_motionArguments", "MotionResponse"),
        "Error executing tool set_motion: connection reset",
        "Error executing tool set_motion: internal error",
    ]:
        event["item"]["result"]["content"][0]["text"] = replacement
        path.write_text(json.dumps(event))
        assert len(result_errors(result)) == 1
    event["item"]["result"]["content"][0]["text"] = text
    event["item"]["error"] = {"message": "transport failed"}
    path.write_text(json.dumps(event))
    assert len(result_errors(result)) == 1


def test_trolley_approval_requires_exact_private_local_command():
    from drivingbench.replay.clients import is_offline_replay

    command = [
        sys.executable,
        "-m",
        "drivingbench.replay.trolley",
        "--scenario",
        "/tmp/scenario.json",
        "--output",
        "/tmp/trolley-output",
    ]
    assert is_offline_replay(command)
    assert not is_offline_replay(command + ["--gateway-url", "http://127.0.0.1"])
    for index, value in [
        (0, "python"),
        (2, "drivingbench.mcp.tools"),
        (4, "relative.json"),
        (4, "/tmp/scenario.yaml"),
        (6, "relative-output"),
    ]:
        changed = command.copy()
        changed[index] = value
        assert not is_offline_replay(changed)


def private_command():
    return [
        sys.executable,
        "-m",
        "drivingbench.replay.mcp",
        "--gateway-url",
        "http://127.0.0.1:8123",
    ]


def test_normal_codex_preserves_settings_and_replaces_renamed_transports(tmp_path, monkeypatch):
    source = tmp_path / ".codex"
    source.mkdir()
    original = """approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "live"
[features]
shell_tool = true
[mcp_servers.notes]
url = "https://notes.example/mcp"
bearer_token_env_var = "NOTES_TOKEN"
[mcp_servers.unknown_alias]
command = "/wrapper"
args = ["--endpoint", "http://127.0.0.1:8766"]
env = { SECRET = "never-log-this" }
[mcp_servers.drivingbench_sandbox]
url = "http://car:8876"
[profiles.personal.mcp_servers.renamed]
command = "/venv/bin/drivingbench-sandbox"
"""
    (source / "config.toml").write_text(original)
    (source / "AGENTS.md").write_text("Usual instructions")
    (source / "sessions").mkdir()
    (source / "sessions/private-chat.jsonl").write_text("chat should not be copied")
    monkeypatch.setenv("CODEX_APP_TOOLS_PIPE_PATH", "/parent/task/socket")
    client = prepare_client(
        CLIENTS["astra"],
        tmp_path / "state",
        private_command(),
        "scene",
        auth_home=source,
        profile="normal",
    )
    config = tomllib.loads((client.state_dir / "codex/config.toml").read_text())
    assert config["sandbox_mode"] == "danger-full-access"
    assert config["web_search"] == "live" and config["features"]["shell_tool"]
    assert config["mcp_servers"]["notes"] == tomllib.loads(original)["mcp_servers"]["notes"]
    entry = config["mcp_servers"]["scene"]
    assert config["mcp_servers"]["unknown_alias"] == entry
    assert config["mcp_servers"]["drivingbench_sandbox"] == entry
    assert config["profiles"]["personal"]["mcp_servers"]["renamed"] == entry
    assert entry["tools"]["observe"] == {"approval_mode": "approve"}
    assert (source / "config.toml").read_text() == original
    assert (client.state_dir / "codex/AGENTS.md").read_text() == "Usual instructions"
    assert not (client.state_dir / "codex/sessions").exists()
    assert "CODEX_APP_TOOLS_PIPE_PATH" not in client.env
    assert "--ignore-rules" not in client.command("drive")
    inventory = (client.state_dir / "tool-inventory.json").read_text()
    assert "never-log-this" not in inventory and "notes.example" not in inventory
    # Resuming reuses the frozen settings snapshot, not a newly changed normal config.
    (source / "config.toml").write_text('web_search="disabled"')
    again = prepare_client(
        CLIENTS["astra"],
        tmp_path / "state",
        private_command(),
        "scene",
        auth_home=source,
        profile="normal",
    )
    assert again.cwd == client.cwd
    assert (
        tomllib.loads((client.state_dir / "codex/config.toml").read_text())["web_search"] == "live"
    )


@pytest.mark.parametrize("family,model", [("claude", "fable"), ("cursor", "grok")])
def test_normal_json_clients_preserve_tools_and_never_disable_unrelated(
    tmp_path, monkeypatch, family, model
):
    source = tmp_path / ("." + family)
    source.mkdir()
    servers = {
        "drivingbench_sandbox": {"url": "http://car:8877", "headers": {"secret": "private"}},
        "rename": {"command": "/python", "args": ["-m", "drivingbench.mcp.tools"]},
        "notes": {"url": "https://notes.example/mcp"},
    }
    path = source.parent / ".claude.json" if family == "claude" else source / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers, "projects": {"private-chat": "not copied"}}))
    settings_name = "settings.json" if family == "claude" else "cli-config.json"
    settings = {
        "permissions": {"allow": ["Read(*)"], "deny": []},
        "model": "normal-model",
        "unrecognized": {"preserve": True},
    }
    (source / settings_name).write_text(json.dumps(settings))
    before = {p: p.read_bytes() for p in source.parent.rglob("*.json")}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("No CLI calls in normal preparation")
    )
    client = prepare_client(
        CLIENTS[model],
        tmp_path / "state",
        private_command(),
        "scene",
        auth_home=source,
        profile="normal",
    )
    config_dir = Path(
        client.env["CLAUDE_CONFIG_DIR" if family == "claude" else "CURSOR_CONFIG_DIR"]
    )
    prepared_settings = json.loads((config_dir / settings_name).read_text())
    expected_settings = json.loads(json.dumps(settings))
    if family == "cursor":
        expected_settings["permissions"]["allow"] += [
            f"Mcp({alias}:{tool})"
            for alias in ("drivingbench_sandbox", "rename", "scene")
            for tool in ("observe", "set_motion", "stop_now")
        ]
    assert prepared_settings == expected_settings
    mcp = json.loads((client.state_dir / "mcp.json").read_text())["mcpServers"]
    assert mcp["notes"] == servers["notes"]
    assert mcp["scene"] == mcp["rename"] == mcp["drivingbench_sandbox"]
    assert "private-chat" not in (client.state_dir / "mcp.json").read_text()
    assert all(p.read_bytes() == data for p, data in before.items())
    if family == "claude":
        command = client.command("observe")
        assert "--strict-mcp-config" in command
        assert not {"--tools", "--setting-sources", "--permission-mode", "--no-chrome"} & set(
            command
        )
        allowed = set(command[command.index("--allowedTools") + 1].split(","))
        assert "Read(*)" in allowed
        for alias in ("scene", "rename", "drivingbench_sandbox"):
            assert {
                f"mcp__{alias}__{tool}" for tool in ("observe", "set_motion", "stop_now")
            } <= allowed
        assert "mcp__notes__*" not in allowed and "Bash(*)" not in allowed
        assert command[-2:] == ["--", "observe"]
    else:
        assert json.loads((client.cwd / ".cursor/mcp.json").read_text())["mcpServers"] == mcp


def test_normal_profile_rejects_physical_command_and_detects_tampering(tmp_path):
    command = private_command()
    command[-1] = "http://127.0.0.1:8766"
    with pytest.raises(ValueError, match="physical"):
        prepare_client(CLIENTS["astra"], tmp_path / "bad", command, profile="normal")
    source = tmp_path / ".codex"
    source.mkdir()
    client = prepare_client(
        CLIENTS["astra"], tmp_path / "state", private_command(), auth_home=source, profile="normal"
    )
    path = client.state_dir / "codex/config.toml"
    path.write_text(path.read_text() + '\n[mcp_servers.sneaky]\nurl="http://127.0.0.1:8877"\n')
    with pytest.raises(ValueError, match="survived"):
        client.start("drive", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_linked_plugin_physical_endpoint_blocks_before_any_process(tmp_path):
    source = tmp_path / ".codex"
    plugin = source / "plugins/cache/example"
    plugin.mkdir(parents=True)
    (plugin / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"car_alias": {"url": "http://car:8766"}}})
    )
    with pytest.raises(ValueError, match="asset contains"):
        prepare_client(
            CLIENTS["astra"],
            tmp_path / "state",
            private_command(),
            auth_home=source,
            profile="normal",
        )


def test_bare_physical_port_and_direct_plugin_map_are_detected(tmp_path):
    assert clients.driving_endpoint("renamed", {"command": "/opaque", "args": ["--port", "8766"]})
    assert clients.driving_endpoint("renamed", {"command": "/opaque", "env": {"PORT": 8876}})
    assert not clients.driving_endpoint(
        "notes", {"command": "/notes", "description": "DrivingBench notes on port 8766"}
    )
    source = tmp_path / ".codex"
    plugin = source / "plugins/cache/example"
    plugin.mkdir(parents=True)
    (plugin / ".mcp.json").write_text(
        json.dumps(
            {"unknown_alias": {"command": "/python", "args": ["-m", "drivingbench.mcp.tools"]}}
        )
    )
    with pytest.raises(ValueError, match="asset contains"):
        prepare_client(
            CLIENTS["astra"],
            tmp_path / "state",
            private_command(),
            auth_home=source,
            profile="normal",
        )


@pytest.mark.parametrize("manifest_dir", [".codex-plugin", ".cursor-plugin", ".claude-plugin"])
@pytest.mark.parametrize("as_list", [False, True])
def test_plugin_mcp_file_reference_is_resolved_from_plugin_root(tmp_path, manifest_dir, as_list):
    source = tmp_path / ".codex"
    plugin = source / "plugins/cache/example"
    (plugin / manifest_dir).mkdir(parents=True)
    # A nonstandard filename makes reference following necessary, not just recursive discovery.
    reference = "./definitions.json"
    (plugin / manifest_dir / "plugin.json").write_text(
        json.dumps({"mcpServers": [reference] if as_list else reference})
    )
    endpoint = plugin / "definitions.json"
    harmless = {"mcpServers": {"notes": {"url": "https://notes.example/mcp"}}}
    endpoint.write_text(json.dumps(harmless))
    client = prepare_client(
        CLIENTS["astra"], tmp_path / "state", private_command(), auth_home=source, profile="normal"
    )
    assert clients.assert_private_profile(client)
    assert json.loads(endpoint.read_text()) == harmless
    endpoint.write_text(json.dumps({"mcpServers": {"renamed": {"url": "http://127.0.0.1:8766"}}}))
    with pytest.raises(ValueError, match="asset contains"):
        clients.assert_private_profile(client)


def test_missing_plugin_mcp_reference_blocks_preparation(tmp_path):
    source = tmp_path / ".codex"
    manifest = source / "plugins/cache/example/.codex-plugin"
    manifest.mkdir(parents=True)
    (manifest / "plugin.json").write_text('{"mcpServers":"./missing.json"}')
    with pytest.raises(ValueError, match="reference does not exist"):
        prepare_client(
            CLIENTS["astra"],
            tmp_path / "state",
            private_command(),
            auth_home=source,
            profile="normal",
        )


def test_refresh_timeout_cancels_group_once_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        clients,
        "claude_oauth",
        lambda: {"accessToken": "old", "refreshToken": "valid", "expiresAt": 1},
    )
    calls = []

    class Process:
        pid = 12345678
        returncode = None

        def wait(self, timeout=None):
            if timeout == 60:
                raise subprocess.TimeoutExpired("fake-claude", timeout)
            self.returncode = -15
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    audit = clients.refresh_claude_credentials(CLIENTS["fable"], tmp_path / "refresh")
    assert audit["timed_out"] and audit["model_calls"] == 1
    assert audit["returncode"] == -15 and len(calls) == 1
    assert not audit["access_current"] and not audit["expiry_advanced"]


def test_refresh_helper_calls_normal_cli_once_and_never_logs_tokens(tmp_path, monkeypatch):
    credentials = iter(
        [
            {"accessToken": "secret-old", "refreshToken": "secret-refresh", "expiresAt": 1},
            {
                "accessToken": "secret-new",
                "refreshToken": "secret-refresh",
                "expiresAt": (time.time() + 5000) * 1000,
            },
        ]
    )
    monkeypatch.setattr(clients, "claude_oauth", lambda: next(credentials))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "inherited-stale-token")
    monkeypatch.setenv("CODEX_APP_TOOLS_PIPE_PATH", "/parent")
    calls = []

    class Process:
        returncode = 0

        def wait(self, timeout):
            assert timeout == 60

    def launch(command, **kwargs):
        calls.append(command)
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'
        assert not any(
            k.startswith(("CODEX_", "CLAUDE_", "CURSOR_", "ANTHROPIC_")) for k in kwargs["env"]
        )
        assert not Path(kwargs["cwd"]).is_relative_to(tmp_path)
        kwargs["stdout"].write(b'{"result":"OK","total_cost_usd":0.08}')
        return Process()

    monkeypatch.setattr(subprocess, "Popen", launch)
    audit = clients.refresh_claude_credentials(CLIENTS["fable"], tmp_path / "refresh")
    assert len(calls) == 1 and audit["model_calls"] == 1 and audit["expiry_advanced"]
    assert audit["access_current"] and "secret" not in json.dumps(audit)
    assert (tmp_path / "refresh/stdout.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        clients.refresh_claude_credentials(CLIENTS["fable"], tmp_path / "refresh")


@pytest.mark.parametrize(
    "oauth,action",
    [({}, "refresh_unavailable"), ({"accessToken": "fresh", "expiresAt": 10**15}, "not_needed")],
)
def test_refresh_helper_does_not_call_for_fresh_or_unrefreshable_auth(
    tmp_path, monkeypatch, oauth, action
):
    monkeypatch.setattr(clients, "claude_oauth", lambda: oauth)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("No model call expected"))
    audit = clients.refresh_claude_credentials(CLIENTS["fable"], tmp_path / "refresh")
    assert audit["action"] == action and audit["model_calls"] == 0


def test_normal_still_profile_preapproves_only_observe(tmp_path):
    source = tmp_path / ".claude"
    source.mkdir()
    command = [
        sys.executable,
        "-m",
        "drivingbench.replay.fork_say",
        "serve",
        "--image",
        "/tmp/frame.jpg",
    ]
    client = prepare_client(
        CLIENTS["fable"], tmp_path / "state", command, "scene", auth_home=source, profile="normal"
    )
    args = client.command("classify")
    assert args[args.index("--allowedTools") + 1] == "mcp__scene__observe"


@pytest.mark.parametrize("image_only", [False, True])
def test_normal_cursor_adds_only_declared_private_mcp_permissions(tmp_path, image_only):
    source = tmp_path / ".cursor"
    source.mkdir()
    settings = {
        "approvalMode": "allowlist",
        "permissions": {
            "allow": ["Shell(git status)", "Mcp(notes:read)"],
            "deny": ["Read(/private/secrets)", "Mcp(notes:write)"],
        },
    }
    (source / "cli-config.json").write_text(json.dumps(settings))
    (source / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "drivingbench_sandbox": {"url": "http://127.0.0.1:8766"},
                    "renamed_car": {"command": "/venv/bin/drivingbench-sandbox"},
                    "notes": {"url": "https://notes.example/mcp"},
                }
            }
        )
    )
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    command = (
        [sys.executable, "-m", "drivingbench.replay.fork_say", "serve", "--image", "/tmp/frame.jpg"]
        if image_only
        else private_command()
    )
    client = prepare_client(
        CLIENTS["grok"],
        tmp_path / "state",
        command,
        "scene" if image_only else "drivingbench_realvehicle",
        auth_home=source,
        profile="normal",
    )
    saved = json.loads((Path(client.env["CURSOR_CONFIG_DIR"]) / "cli-config.json").read_text())
    aliases = {"drivingbench_sandbox", "renamed_car", client.server_name}
    tools = {"observe"} if image_only else {"observe", "set_motion", "stop_now"}
    assert set(saved["permissions"]["allow"]) == set(settings["permissions"]["allow"]) | {
        f"Mcp({alias}:{tool})" for alias in aliases for tool in tools
    }
    assert saved["permissions"]["deny"] == settings["permissions"]["deny"]
    assert saved["approvalMode"] == "allowlist"
    assert not {"--force", "--yolo", "--mode"} & set(client.command("drive"))
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


@pytest.mark.parametrize("family", ["codex", "cursor", "claude"])
def test_normal_errors_keep_driving_denials_but_off_task_denials_are_behavior(tmp_path, family):
    if family == "codex":
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "id": str(i),
                    "server": server,
                    "tool": "observe",
                    "status": "failed",
                    "error": f"{server} permission denied",
                },
            }
            for i, server in enumerate(("notes", "private_car"))
        ]
    elif family == "cursor":
        events = [
            {
                "type": "tool_call",
                "subtype": "completed",
                "call_id": str(i),
                "tool_call": {
                    "mcpToolCall": {
                        "args": {"serverIdentifier": server, "toolName": "observe", "args": {}},
                        "result": {"permissionDenied": f"{server} denied"},
                    }
                },
            }
            for i, server in enumerate(("notes", "private_car"))
        ]
    else:
        events = [
            {
                "type": "result",
                "permission_denials": [{"tool_name": "Read", "tool_input": {"path": "/notes"}}],
            },
            {
                "type": "result",
                "permission_denials": [
                    {"tool_name": "mcp__private_car__observe", "tool_input": {}}
                ],
            },
        ]
    path = tmp_path / "stdout.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    assert len(clients.result_errors(result)) == 2
    assert len(clients.result_errors(result, driving_servers={"private_car"})) == 1
    path.write_text(json.dumps(events[0]))
    assert clients.result_errors(result, driving_servers={"private_car"}) == []


def test_normal_error_filter_preserves_cli_api_and_unknown_failures(tmp_path):
    events = [
        {
            "type": "item.completed",
            "item": {"type": "mcp_tool_call", "status": "failed", "error": "unknown target"},
        },
        {"type": "error", "error": "401 unauthorized"},
        {"type": "result", "is_error": True, "result": "API timeout"},
        {"type": "result", "permission_denials": [{"unknown_format": "denied"}]},
    ]
    path = tmp_path / "stdout.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    result = ClientResult(1, "id", path, tmp_path / "stderr")
    errors = clients.result_errors(result, driving_servers={"private_car"})
    assert len(errors) == 5 and "client_exit_1" in errors
    assert "401 unauthorized" in errors and "API timeout" in errors


def test_claude_denial_identity_joins_discovery_intent(tmp_path):
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mail",
                        "name": "ToolSearch",
                        "input": {"query": "select:mcp__mail__send"},
                    },
                    {
                        "type": "tool_use",
                        "id": "car",
                        "name": "ToolSearch",
                        "input": {"query": "select:mcp__private_car__observe"},
                    },
                ]
            },
        },
        {"type": "result", "permission_denials": [{"tool_use_id": "mail"}, {"tool_use_id": "car"}]},
    ]
    path = tmp_path / "stdout.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events))
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    errors = clients.result_errors(result, driving_servers={"private_car"})
    assert len(errors) == 1 and "car" in errors[0] and "mail" not in errors[0]


def test_missing_mcp_server_identity_cannot_hide_a_driving_failure(tmp_path):
    path = tmp_path / "stdout.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "observe",
                    "status": "failed",
                    "error": "connection reset",
                },
            }
        )
    )
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    assert clients.result_errors(result, driving_servers={"private_car"}) == [
        "mcp_tool_failure: connection reset"
    ]


def test_off_task_call_error_flag_does_not_hide_explicit_api_error(tmp_path):
    path = tmp_path / "stdout.jsonl"
    event = {
        "type": "item.completed",
        "is_error": True,
        "item": {
            "type": "mcp_tool_call",
            "server": "notes",
            "tool": "read",
            "status": "failed",
            "error": "denied",
        },
    }
    path.write_text(json.dumps(event))
    result = ClientResult(0, "id", path, tmp_path / "stderr")
    assert clients.result_errors(result, driving_servers={"private_car"}) == []
    event["is_api_error_message"] = True
    path.write_text(json.dumps(event))
    assert clients.result_errors(result, driving_servers={"private_car"})


@pytest.mark.parametrize(
    "model,family", [("astra", "codex"), ("fable", "claude"), ("grok", "cursor")]
)
def test_normal_profile_can_consolidate_only_driving_aliases(tmp_path, monkeypatch, model, family):
    source = tmp_path / ("." + family)
    source.mkdir()
    servers = {
        "drivingbench_sandbox": {"url": "http://car:8877"},
        "drivingbench_realvehicle": {
            "command": "/python",
            "args": ["-m", "drivingbench.mcp.tools"],
        },
        "notes": {"url": "https://notes.example/mcp"},
    }
    if family == "codex":
        import tomlkit

        document = dict(
            sandbox_mode="danger-full-access",
            web_search="live",
            mcp_servers=servers,
            profiles=dict(personal=dict(mcp_servers=dict(hidden=servers["drivingbench_sandbox"]))),
        )
        (source / "config.toml").write_text(tomlkit.dumps(document))
    else:
        path = source.parent / ".claude.json" if family == "claude" else source / "mcp.json"
        path.write_text(json.dumps({"mcpServers": servers}))
        name = "settings.json" if family == "claude" else "cli-config.json"
        (source / name).write_text(
            json.dumps(
                dict(
                    permissions=dict(
                        allow=["Read(*)", "Mcp(notes:read)", "mcp__drivingbench_sandbox__observe"],
                        deny=["Mcp(notes:write)"],
                    ),
                    ordinary_setting=True,
                )
            )
        )
    originals = {p: p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: pytest.fail("No CLI invocation during profile preparation"),
    )
    client = prepare_client(
        CLIENTS[model],
        tmp_path / "state",
        private_command(),
        "drivingbench",
        auth_home=source,
        profile="normal",
        consolidate_driving_aliases=True,
    )
    rows = clients.assert_private_profile(client)
    assert {r["name"] for r in rows if r["private_driving"]} == {"drivingbench"}
    assert {r["name"] for r in rows if not r["private_driving"]} == {"notes"}
    assert all(p.read_bytes() == data for p, data in originals.items())
    if family == "codex":
        config = tomllib.loads((client.state_dir / "codex/config.toml").read_text())
        assert config["web_search"] == "live"
        assert config["mcp_servers"]["notes"] == servers["notes"]
    else:
        config_dir = Path(
            client.env["CLAUDE_CONFIG_DIR" if family == "claude" else "CURSOR_CONFIG_DIR"]
        )
        settings = json.loads((config_dir / name).read_text())
        assert settings["ordinary_setting"]
        assert "Read(*)" in settings["permissions"]["allow"]
        assert settings["permissions"]["deny"] == ["Mcp(notes:write)"]
        assert "drivingbench_sandbox" not in json.dumps(settings)
        config = json.loads((client.state_dir / "mcp.json").read_text())
        assert config["mcpServers"]["notes"] == servers["notes"]
    if family == "claude":
        command = client.command("drive")
        allowed = command[command.index("--allowedTools") + 1]
        assert "mcp__drivingbench__observe" in allowed
        assert "drivingbench_sandbox" not in allowed and "drivingbench_realvehicle" not in allowed
        assert "--tools" not in command
    with pytest.raises(ValueError, match="Cannot change"):
        prepare_client(
            CLIENTS[model],
            tmp_path / "state",
            private_command(),
            "drivingbench",
            auth_home=source,
            profile="normal",
        )


def realism_command(gateway="http://127.0.0.1:8123"):
    from drivingbench.sim.studies.persistent_goal_mcp import configuration

    return configuration({"arm": "straight_realism"}, gateway)[0]


@pytest.mark.parametrize("gateway", ["http://127.0.0.1:8123", "http://127.0.0.1:8123/"])
def test_realism_private_admission_requires_exact_dedicated_transport(gateway):
    command = realism_command(gateway)
    assert clients.is_offline_replay(command)
    clients._private_command(command)
    for changed in (
        command + ["--client", "other"],
        command[:-1],
        ["python", *command[1:]],
        [*command[:5], "--realism", "--realism"],
        [*command[:3], "--realism", "--gateway-url", gateway],
    ):
        assert not clients.is_offline_replay(changed)
        with pytest.raises(ValueError, match="recognized private replay"):
            clients._private_command(changed)


@pytest.mark.parametrize(
    "gateway",
    [
        "http://127.0.0.1:8766",
        "http://127.0.0.1:8876",
        "http://127.0.0.1:8877",
        "http://127.0.0.1",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:bad",
        "http://localhost:8123",
        "http://[::1]:8123",
        "http://192.168.1.10:8123",
        "https://127.0.0.1:8123",
        "http://user:secret@127.0.0.1:8123",
        "http://127.0.0.1:8123/path",
        "http://127.0.0.1:8123?token=secret",
        "http://127.0.0.1:8123#fragment",
    ],
)
def test_realism_private_admission_rejects_other_endpoints_before_state_creation(tmp_path, gateway):
    command = realism_command(gateway)
    assert not clients.is_offline_replay(command)
    state = tmp_path / "private-client-state"
    with pytest.raises(ValueError, match="recognized private replay"):
        prepare_client(CLIENTS["astra"], state, command, profile="normal")
    assert not state.exists()


@pytest.mark.parametrize("model", ["astra", "sol", "fable", "grok"])
def test_actual_normal_isolated_realism_preparation(tmp_path, monkeypatch, model):
    """Exercise the real preparer and admission guard without model or authentication calls."""
    from drivingbench.sim.batch import prepare_isolated
    from drivingbench.sim.studies.persistent_goal_mcp import configuration

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_kw: pytest.fail("No process or auth calls in preparation")
    )
    spec = CLIENTS[model]
    source = home / {"codex": ".codex", "claude": ".claude", "cursor": ".cursor"}[spec.family]
    source.mkdir()
    if spec.family == "codex":
        original = 'web_search="live"\n[features]\nshell_tool=true\n[mcp_servers.notes]\nurl="https://notes.example/mcp"\n'
        (source / "config.toml").write_text(original)
    else:
        original = json.dumps({"mcpServers": {"notes": {"url": "https://notes.example/mcp"}}})
        config = home / ".claude.json" if spec.family == "claude" else source / "mcp.json"
        config.write_text(original)
        settings_name = "settings.json" if spec.family == "claude" else "cli-config.json"
        (source / settings_name).write_text(json.dumps({"permissions": {"allow": ["Read(*)"]}}))
    record = tmp_path / "cell"
    record.mkdir()
    command, alias = configuration({"arm": "straight_realism"}, "http://127.0.0.1:8123")
    client = prepare_isolated(
        spec,
        record / "private-client-state",
        command,
        record,
        server_name=alias,
        profile="normal",
        consolidate_driving_aliases=True,
    )
    assert client is not None and client.profile == "normal"
    assert client.server_name == "drivingbench_realvehicle"
    marker = json.loads((client.state_dir / "client-profile.json").read_text())
    assert marker["private_tools"] == ["observe", "set_motion", "stop_now"]
    launcher = Path(marker["private_entry"]["command"])
    assert "drivingbench.sim.studies.persistent_goal_mcp" in launcher.read_text()
    assert "--realism" in launcher.read_text()
    assert list(client.cwd.iterdir()) == (
        [] if spec.family != "cursor" else [client.cwd / ".cursor"]
    )
    assert not list(record.glob("credential-attempt-*.json"))
    if spec.family == "codex":
        copied = tomllib.loads((client.state_dir / "codex/config.toml").read_text())
        assert copied["web_search"] == "live" and copied["features"]["shell_tool"]
        assert "notes" in copied["mcp_servers"]
        assert (source / "config.toml").read_text() == original
    else:
        copied = json.loads((client.state_dir / "mcp.json").read_text())
        assert "notes" in copied["mcpServers"]
        assert config.read_text() == original
