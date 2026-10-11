"""Isolated CLI conversations with durable raw output and explicit model selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import tomlkit


@dataclass(frozen=True)
class ClientSpec:
    key: str
    binary: str
    model: str
    family: str
    effort: str = "medium"


CLIENTS = {
    "astra": ClientSpec("astra", "/opt/homebrew/bin/codex", "gpt-6-astra", "codex"),
    "sol": ClientSpec("sol", "/opt/homebrew/bin/codex", "gpt-5.6-sol", "codex"),
    "fable": ClientSpec(
        "fable", str(Path.home() / ".local/bin/claude"), "claude-fable-5-1", "claude"
    ),
    "grok": ClientSpec(
        "grok", str(Path.home() / ".local/bin/agent"), "cursor-grok-4.6-medium", "cursor"
    ),
}
STATE_ROOT = Path.home() / ".cache/drivingbench-client-state"


def read_events(path: Path) -> list[dict]:
    events = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    events.append(value)
            except ValueError:
                pass
    return events


@dataclass(frozen=True)
class ClientResult:
    returncode: int
    session_id: str | None
    stdout_path: Path
    stderr_path: Path


def _tool_input_validation(item: dict) -> str | None:
    """Recognize the MCP argument-schema error, never transport or server exceptions."""
    tool = item.get("tool")
    if (
        item.get("type") != "mcp_tool_call"
        or item.get("status") != "failed"
        or item.get("error") is not None
        or tool not in {"observe", "set_motion", "stop_now"}
    ):
        return None
    content = (item.get("result") or {}).get("content", [])
    if len(content) != 1 or content[0].get("type") != "text":
        return None
    text = content[0].get("text", "")
    prefix = rf"Error executing tool {re.escape(tool)}: [1-9][0-9]* validation errors? for {re.escape(tool)}Arguments\n"
    if re.match(prefix, text) and "[type=" in text and "https://errors.pydantic.dev/" in text:
        return text
    return None


def input_validation_errors(result: ClientResult) -> list[dict]:
    """Keep model-authored invalid calls visible without mislabeling them infrastructure."""
    errors = []
    for index, event in enumerate(read_events(result.stdout_path), 1):
        item = event.get("item", {})
        detail = _tool_input_validation(item)
        if detail:
            errors.append(
                {
                    "event_index": index,
                    "tool": item["tool"],
                    "arguments": item.get("arguments"),
                    "error": detail,
                    "source": str(result.stdout_path),
                }
            )
    return errors


def result_errors(result: ClientResult, *, driving_servers=None) -> list[str]:
    """Keep infrastructure failures; optionally treat identifiable off-task failures as behavior."""
    events = read_events(result.stdout_path)
    activity = []
    if driving_servers is not None:
        from drivingbench.replay.tool_activity import tool_activity

        driving_servers = set(driving_servers)
        activity = tool_activity(events, driving_servers=driving_servers)

    def off_task(rows):
        return bool(rows) and all(row["category"] == "off_task" for row in rows)

    def off_task_denial(denial):
        if driving_servers is None or not isinstance(denial, dict):
            return False
        identity = denial.get("tool_use_id") or denial.get("toolUseId") or denial.get("id")
        rows = [row for row in activity if row["call_id"] == identity] if identity else []
        if rows:
            return off_task(rows)
        name = denial.get("tool_name") or denial.get("toolName") or denial.get("name")
        if not isinstance(name, str) or not name:
            return False
        event = {
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": name,
                        "input": denial.get("tool_input", denial.get("input")),
                    }
                ]
            }
        }
        return off_task(tool_activity([event], driving_servers=driving_servers))

    errors = []
    if result.returncode:
        errors.append(f"client_exit_{result.returncode}")
    for index, event in enumerate(events, 1):
        behavior_failure = off_task([row for row in activity if index in row["event_indices"]])
        if (
            (event.get("is_error") and not behavior_failure)
            or event.get("is_api_error_message")
            or event.get("type") in {"error", "turn.failed"}
            or event.get("subtype", "").startswith("error")
        ):
            errors.append(str(event.get("result") or event.get("error") or event))
        item = event.get("item", {})
        if (
            item.get("type") == "mcp_tool_call"
            and item.get("status") == "failed"
            and _tool_input_validation(item) is None
            and not behavior_failure
        ):
            errors.append(f"mcp_tool_failure: {item.get('error')}")
        mcp_result = event.get("tool_call", {}).get("mcpToolCall", {}).get("result", {})
        for key in ("rejected", "permissionDenied", "error"):
            if key in mcp_result and not behavior_failure:
                errors.append(f"mcp_tool_failure: {mcp_result[key]}")
        if event.get("type") == "result" and event.get("permission_denials"):
            denials = [
                denial for denial in event["permission_denials"] if not off_task_denial(denial)
            ]
            if denials:
                errors.append(f"mcp_permission_denials: {denials}")
    return list(dict.fromkeys(errors))


def extract_messages(result: ClientResult) -> list[dict]:
    """Normalize visible assistant text with auditable raw stream locations."""
    messages = []
    for line_number, line in enumerate(
        result.stdout_path.read_text(errors="replace").splitlines(), 1
    ):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item", {})
        text = ""
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text", "")
        elif event.get("type") == "assistant" and not event.get("is_api_error_message"):
            content = event.get("message", {}).get("content", [])
            text = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        if text:
            timestamp = event.get("timestamp")
            if timestamp is None and isinstance(event.get("timestamp_ms"), (int, float)):
                timestamp = datetime.fromtimestamp(event["timestamp_ms"] / 1000, UTC).isoformat()
            source = f"{result.stdout_path}:{line_number}"
            messages.append(
                {
                    "id": source,
                    "source": source,
                    "role": "assistant",
                    "text": text,
                    "timestamp": timestamp,
                }
            )
    return messages


@dataclass
class RunningClient:
    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path

    def poll(self) -> int | None:
        return self.process.poll()

    def turn_done(self) -> bool:
        """The turn ended in the stream; Codex can linger after `turn.completed`."""
        return self.poll() is not None or any(
            e.get("type") in ("turn.completed", "turn.failed", "result")
            for e in read_events(self.stdout_path)
        )

    def cancel(self) -> None:
        """Stop the whole CLI/MCP process group, escalating after a short grace period."""
        if self.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):  # macOS: EPERM for an exited group
                return
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()

    def finish(self, timeout: float | None = None) -> ClientResult:
        code = self.process.wait(timeout=timeout)
        session = None
        for event in read_events(self.stdout_path):
            session = event.get("thread_id") or event.get("session_id") or session
        return ClientResult(code, session, self.stdout_path, self.stderr_path)


@dataclass
class PreparedClient:
    spec: ClientSpec
    state_dir: Path
    cwd: Path
    env: dict[str, str]
    server_name: str
    profile: str = "restricted"

    def command(self, prompt: str, session_id: str | None = None) -> list[str]:
        spec = self.spec
        if spec.family == "codex":
            args = [spec.binary, "exec"]
            if session_id:
                args += ["resume", session_id]
            args += ["--json", "--skip-git-repo-check", "-m", spec.model]
            if self.profile == "restricted":
                args += ["--ignore-rules"]
            args += ["-c", f'model_reasoning_effort="{spec.effort}"']
        elif spec.family == "claude":
            args = [
                spec.binary,
                "-p",
                "--verbose",
                "--output-format",
                "stream-json",
                "--model",
                spec.model,
                "--effort",
                spec.effort,
                "--strict-mcp-config",
                "--mcp-config",
                str(self.state_dir / "mcp.json"),
            ]
            if self.profile == "restricted":
                args += [
                    "--setting-sources",
                    "",
                    "--tools",
                    "",
                    "--permission-mode",
                    "dontAsk",
                    "--allowedTools",
                    f"mcp__{self.server_name}__*",
                    "--no-chrome",
                ]
            else:
                profile = _json(self.state_dir / "client-profile.json")
                aliases = (
                    {self.server_name}
                    if profile.get("consolidate_driving_aliases")
                    else set(profile["replaced_servers"]) | {self.server_name}
                )
                allowed = []
                for name in ("settings.json", "settings.local.json"):
                    allowed.extend(
                        _json(Path(self.env["CLAUDE_CONFIG_DIR"]) / name)
                        .get("permissions", {})
                        .get("allow", [])
                    )
                allowed.extend(
                    f"mcp__{alias}__{tool}"
                    for alias in sorted(aliases)
                    for tool in profile["private_tools"]
                )
                args += ["--allowedTools", ",".join(dict.fromkeys(allowed))]
            if session_id:
                args += ["--resume", session_id]
            if self.profile == "normal":
                args += ["--"]
        elif spec.family == "cursor":
            args = [
                spec.binary,
                "-p",
                "--output-format",
                "stream-json",
                "--model",
                spec.model,
                "--approve-mcps",
                "--trust",
                "--workspace",
                str(self.cwd),
            ]
            if session_id:
                args += ["--resume", session_id]
        else:
            raise ValueError(f"Unknown client family: {spec.family}")
        return args + [prompt]

    def start(
        self, prompt: str, output_dir: Path, session_id: str | None = None, *, wall_deadline=None
    ) -> RunningClient:
        if wall_deadline is not None and (
            type(wall_deadline) not in (int, float) or not math.isfinite(wall_deadline)
        ):
            raise ValueError("Finite private launch deadline required")
        if self.profile == "normal":
            assert_private_profile(self)
        args = self.command(prompt, session_id)
        version = subprocess.run(
            [self.spec.binary, "--version"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout = output_dir / "stdout.jsonl"
        stderr = output_dir / "stderr.log"
        if stdout.exists() or stderr.exists():
            raise FileExistsError("Refusing to overwrite a conversation's outputs")
        (output_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "client": self.spec.key,
                    "version": version,
                    "binary": self.spec.binary,
                    "model": self.spec.model,
                    "effort": self.spec.effort,
                    "tool_profile": self.profile,
                    "session_id": session_id,
                    "prompt": prompt,
                    "command": args,
                    "cwd": str(self.cwd),
                },
                indent=2,
            )
        )
        with stdout.open("wb") as out, stderr.open("wb") as err:
            if wall_deadline is not None and time.monotonic() >= wall_deadline:
                raise TimeoutError("Original drive deadline before CLI Popen")
            process = subprocess.Popen(
                args,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        return RunningClient(process, stdout, stderr)


def is_offline_replay(command: list[str]) -> bool:
    """Preauthorize only private loopback replay or the exact local trolley entry point."""
    if (
        len(command) == 7
        and Path(command[0]).is_absolute()
        and command[1:3] == ["-m", "drivingbench.replay.trolley"]
        and command[3] == "--scenario"
        and command[5] == "--output"
    ):
        return (
            Path(command[4]).is_absolute()
            and Path(command[4]).suffix == ".json"
            and Path(command[6]).is_absolute()
        )
    if (
        len(command) == 8
        and Path(command[0]).is_absolute()
        and command[1:4] == ["-m", "drivingbench.replay.fork", "proxy"]
        and command[4] == "--scenario"
        and command[6] == "--output"
    ):
        return (
            Path(command[5]).is_absolute()
            and Path(command[5]).suffix == ".json"
            and Path(command[7]).is_absolute()
        )
    if (
        len(command) == 6
        and Path(command[0]).is_absolute()
        and command[1:5] == ["-m", "drivingbench.replay.fork_say", "serve", "--image"]
    ):
        return Path(command[5]).is_absolute() and Path(command[5]).suffix == ".jpg"
    if (
        len(command) == 6
        and Path(command[0]).is_absolute()
        and command[1:5] == ["-m", "drivingbench.sim.perception_probe", "serve", "--images"]
    ):
        return Path(command[5]).is_absolute() and Path(command[5]).suffix == ".json"
    if command[1:3] == ["-m", "drivingbench.sim.studies.persistent_goal_mcp"]:
        if (
            len(command) != 6
            or not Path(command[0]).is_absolute()
            or command[3] != "--gateway-url"
            or command[5] != "--realism"
        ):
            return False
        try:
            gateway = urlsplit(command[4])
            return (
                gateway.scheme == "http"
                and gateway.hostname == "127.0.0.1"
                and gateway.port is not None
                and 0 < gateway.port <= 65535
                and gateway.port not in PHYSICAL_PORTS
                and gateway.username is None
                and gateway.password is None
                and gateway.path in {"", "/"}
                and not gateway.query
                and not gateway.fragment
            )
        except ValueError:
            return False
    try:
        module = command[command.index("-m") + 1]
        gateway = urlsplit(command[command.index("--gateway-url") + 1])
    except (ValueError, IndexError):
        return False
    return (
        module == "drivingbench.replay.mcp"
        and gateway.scheme == "http"
        and gateway.hostname in {"127.0.0.1", "::1"}
        and gateway.username is None
        and gateway.password is None
    )


def claude_oauth() -> dict:
    """Claude Code's OAuth entry from the macOS keychain, or {} where there is none."""
    if not shutil.which("security"):
        return {}
    auth = subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
        capture_output=True,
        text=True,
    )
    return json.loads(auth.stdout).get("claudeAiOauth", {}) if auth.returncode == 0 else {}


DRIVING_SERVERS = frozenset(
    {
        "drivingbench_sandbox",
        "drivingbench_realvehicle",
        "drivingbench",
        "drivingbench_sandbox_env",
        "drivingbench_sandbox_fake",
    }
)
PHYSICAL_PORTS = {8766, 8876, 8877}


def driving_endpoint(name: str, entry: dict) -> bool:
    """Recognize renamed DrivingBench transports without logging their credentials."""
    transport = {key: entry[key] for key in ("command", "args", "url", "env") if key in entry}
    text = json.dumps(transport).lower()
    return (
        name.lower() in DRIVING_SERVERS
        or bool(re.search(r"drivingbench(?:[-_](?:sandbox|mcp)|\.(?:mcp|gateway|replay))", text))
        or bool(re.search(r"(?:[:\"\s])(?:8766|8876|8877)(?:[/\s\"'?#,}]|$)", text))
    )


def _private_command(command):
    if not is_offline_replay(command):
        raise ValueError("Normal profile requires a recognized private replay command")
    if "--gateway-url" in command:
        port = urlsplit(command[command.index("--gateway-url") + 1]).port
        if port in PHYSICAL_PORTS:
            raise ValueError("Normal profile cannot use a physical DrivingBench port")


def _replace_driving(document, server, entry, *, consolidate=False):
    """Replace complete definitions at every config nesting level, including profiles."""
    replaced = []
    if not isinstance(document, dict):
        raise ValueError("Client configuration must be an object")
    for key, value in document.items():
        if key in {"mcp_servers", "mcpServers"}:
            if not isinstance(value, dict):
                raise ValueError("MCP server configuration must be an object")
            for name, definition in list(value.items()):
                if not isinstance(definition, dict):
                    raise ValueError("MCP server definition must be an object")
                if name == server or driving_endpoint(name, definition):
                    if consolidate and name != server:
                        del value[name]
                    else:
                        value[name] = dict(entry)
                    replaced.append(name)
        elif isinstance(value, dict):
            replaced.extend(_replace_driving(value, server, entry, consolidate=consolidate))
    return replaced


def _json(path):
    value = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(value, dict):
        raise ValueError("Client configuration must be an object")
    return value


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def _assets(source, target):
    """Reference installed tool resources; snapshot writable metadata, never chat state."""
    references = []
    for name in ("AGENTS.md", "CLAUDE.md", "skills", "rules", "commands", "agents"):
        path = source / name
        if path.exists():
            destination = target / name
            if not destination.exists():
                destination.symlink_to(path.resolve(), target_is_directory=path.is_dir())
            references.append(str(path.resolve()))
    plugins = source / "plugins"
    if plugins.exists():
        destination = target / "plugins"
        destination.mkdir(exist_ok=True)
        for path in plugins.iterdir():
            if path.name in {"cache", "marketplaces", "local"} and path.is_dir():
                if not (destination / path.name).exists():
                    (destination / path.name).symlink_to(path.resolve(), target_is_directory=True)
                references.append(str(path.resolve()))
            elif path.suffix == ".json" and path.is_file():
                shutil.copyfile(path, destination / path.name)
                (destination / path.name).chmod(0o600)
    return references


def _check_asset_endpoints(references):
    checked = set()

    def inspect(path, *, manifest=False):
        path = path.resolve()
        if path in checked:
            return
        checked.add(path)
        if not path.is_file():
            raise ValueError("Installed plugin MCP reference does not exist")
        document = _json(path)
        if manifest:
            root = (
                path.parent.parent
                if path.parent.name in {".codex-plugin", ".cursor-plugin", ".claude-plugin"}
                else path.parent
            )
            for key in ("mcpServers", "mcp_servers"):
                value = document.get(key)
                if isinstance(value, str) or isinstance(value, list):
                    for reference in [value] if isinstance(value, str) else value:
                        if not isinstance(reference, str) or "://" in reference:
                            raise ValueError("Installed plugin MCP reference must be a local file")
                        inspect(root / reference)
                    del document[key]
        elif "mcpServers" not in document and "mcp_servers" not in document:
            document = {"mcpServers": document}
        if _replace_driving(document, "", {}):
            raise ValueError(
                "Installed tool asset contains a DrivingBench endpoint; needs private override"
            )

    for reference in references:
        root = Path(reference)
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.name not in {"mcp.json", ".mcp.json", "plugin.json"}:
                continue
            inspect(path, manifest=path.name == "plugin.json")


def _normal_profile(
    spec, state, cwd, config_dir, source, command, server_name, *, consolidate=False
):
    marker = state / "client-profile.json"
    fingerprint = hashlib.sha256(json.dumps(command).encode()).hexdigest()
    if marker.exists():
        saved = _json(marker)
        if (
            saved["command_sha256"] != fingerprint
            or saved["server_name"] != server_name
            or saved.get("consolidate_driving_aliases", False) != consolidate
        ):
            raise ValueError("Cannot change a prepared normal profile's private transport")
        return
    _private_command(command)
    launcher = config_dir / "private-mcp"
    launcher.write_text("#!/bin/sh\nexec " + shlex.join(command) + "\n")
    launcher.chmod(0o700)
    entry = {"command": str(launcher), "args": []}
    private_tools = (
        ["observe"]
        if any(
            module in command
            for module in ("drivingbench.replay.fork_say", "drivingbench.sim.perception_probe")
        )
        else ["observe", "set_motion", "stop_now"]
    )
    if spec.family == "codex":
        entry["tools"] = {name: {"approval_mode": "approve"} for name in private_tools}
    documents = []
    inputs = []
    if spec.family == "codex":
        path = source / "config.toml"
        document = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()
        inputs.append(path)
        replaced = _replace_driving(document, server_name, entry, consolidate=consolidate)
        document.setdefault("mcp_servers", {})[server_name] = dict(entry)
        document["cli_auth_credentials_store"] = "file"
        target = config_dir / "config.toml"
        target.write_text(tomlkit.dumps(document))
        target.chmod(0o600)
        documents.append(str(target))
    else:
        path = (
            source.parent / (source.name + ".json")
            if spec.family == "claude"
            else source / "mcp.json"
        )
        original = _json(path)
        inputs.append(path)
        # Claude's root file also contains project histories and usage; only MCP settings belong here.
        document = {"mcpServers": original.get("mcpServers", {})}
        replaced = _replace_driving(document, server_name, entry, consolidate=consolidate)
        document["mcpServers"][server_name] = dict(entry)
        private_aliases = [
            name for name, definition in document["mcpServers"].items() if definition == entry
        ]
        targets = [state / "mcp.json"]
        if spec.family == "cursor":
            local = cwd / ".cursor"
            local.mkdir(exist_ok=True)
            targets += [config_dir / "mcp.json", local / "mcp.json"]
        for target in targets:
            _write_json(target, document)
            documents.append(str(target))
        names = (
            ("settings.json", "settings.local.json")
            if spec.family == "claude"
            else ("cli-config.json",)
        )
        for name in names:
            path = source / name
            inputs.append(path)
            settings = _json(path)
            replaced.extend(_replace_driving(settings, server_name, entry, consolidate=consolidate))
            if consolidate:
                permissions = settings.get("permissions", {})
                for key in ("allow", "deny", "ask"):
                    if key not in permissions:
                        continue
                    values = []
                    for value in permissions[key]:
                        for alias in replaced:
                            value = value.replace(f"Mcp({alias}:", f"Mcp({server_name}:").replace(
                                f"mcp__{alias}__", f"mcp__{server_name}__"
                            )
                        values.append(value)
                    permissions[key] = list(dict.fromkeys(values))
            if spec.family == "cursor":
                permissions = settings.setdefault("permissions", {})
                allowed = permissions.setdefault("allow", [])
                permissions["allow"] = list(
                    dict.fromkeys(
                        [
                            *allowed,
                            *(
                                f"Mcp({alias}:{tool})"
                                for alias in private_aliases
                                for tool in private_tools
                            ),
                        ]
                    )
                )
            _write_json(config_dir / name, settings)
            documents.append(str(config_dir / name))
    references = _assets(source, config_dir)
    _check_asset_endpoints(references)
    _write_json(
        marker,
        {
            "profile": "normal",
            "server_name": server_name,
            "consolidate_driving_aliases": consolidate,
            "command_sha256": fingerprint,
            "private_entry": entry,
            "private_tools": private_tools,
            "documents": documents,
            "asset_references": references,
            "replaced_servers": sorted(set(replaced)),
            "sources": [
                {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in inputs
                if p.exists()
            ],
            "scope": "ordinary CLI settings and tools; no parent app IPC or copied chat history",
        },
    )


def assert_private_profile(client):
    """Recheck private config before launch; publish names and booleans, never transports."""
    saved = _json(client.state_dir / "client-profile.json")
    entry = saved["private_entry"]
    launcher = Path(entry["command"])
    command = shlex.split(launcher.read_text().removeprefix("#!/bin/sh\nexec ").strip())
    _private_command(command)
    if hashlib.sha256(json.dumps(command).encode()).hexdigest() != saved["command_sha256"]:
        raise ValueError("Private MCP launcher changed after preparation")
    inventory = []

    def check(document):
        for key, value in document.items():
            if key in {"mcpServers", "mcp_servers"}:
                for name, definition in value.items():
                    private = definition == entry
                    if (
                        private
                        and saved.get("consolidate_driving_aliases")
                        and name != client.server_name
                    ):
                        raise ValueError(
                            "Unexpected private DrivingBench alias after consolidation"
                        )
                    if driving_endpoint(name, definition) and not private:
                        raise ValueError(
                            "Physical DrivingBench endpoint survived private preparation"
                        )
                    inventory.append({"name": name, "private_driving": private})
            elif isinstance(value, dict):
                check(value)

    for name in saved["documents"]:
        path = Path(name)
        check(tomlkit.parse(path.read_text()) if path.suffix == ".toml" else _json(path))
    if not any(row["name"] == client.server_name and row["private_driving"] for row in inventory):
        raise ValueError("Private DrivingBench definition is missing")
    _check_asset_endpoints(saved["asset_references"])
    _write_json(
        client.state_dir / "tool-inventory.json", {"profile": "normal", "servers": inventory}
    )
    return inventory


def refresh_claude_credentials(spec, output, *, timeout_s=60, required_lifetime_s=600):
    """One explicitly authorized maintenance call, using normal keychain auth and no tools."""
    if spec.family != "claude" or not 0 < timeout_s <= 60:
        raise ValueError("Refresh helper requires Claude and a timeout of at most 60 seconds")
    if not math.isfinite(required_lifetime_s) or required_lifetime_s < 0:
        raise ValueError("Required token lifetime must be finite and nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    before = claude_oauth()
    if (
        before.get("accessToken")
        and before.get("expiresAt", 0) / 1000 >= time.time() + required_lifetime_s
    ):
        audit = {"action": "not_needed", "model_calls": 0, "access_current": True}
        _write_json(output / "audit.json", audit)
        return audit
    if not before.get("refreshToken"):
        audit = {
            "action": "refresh_unavailable",
            "model_calls": 0,
            "reason": "no_normal_keychain_refresh_credential",
        }
        _write_json(output / "audit.json", audit)
        return audit
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CODEX_", "CLAUDE_", "CURSOR_", "ANTHROPIC_"))
    }
    command = [
        spec.binary,
        "-p",
        "Reply OK.",
        "--model",
        spec.model,
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--tools",
        "",
        "--setting-sources",
        "",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix="credential-check-") as cwd:
        with (output / "stdout.json").open("wb") as out, (output / "stderr.log").open("wb") as err:
            (output / "stdout.json").chmod(0o600)
            (output / "stderr.log").chmod(0o600)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                RunningClient(process, output / "stdout.json", output / "stderr.log").cancel()
    after = claude_oauth()
    audit = {
        "action": "normal_cli_refresh_maintenance",
        "model_calls": 1,
        "model": spec.model,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_s": time.monotonic() - started,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "before_expires_at_unix_s": before.get("expiresAt", 0) / 1000,
        "after_expires_at_unix_s": after.get("expiresAt", 0) / 1000,
        "expiry_advanced": after.get("expiresAt", 0) > before.get("expiresAt", 0),
        "access_current": bool(after.get("accessToken"))
        and after.get("expiresAt", 0) / 1000 > time.time(),
        "raw_output": "stdout.json",
        "stderr": "stderr.log",
    }
    _write_json(output / "audit.json", audit)
    return audit


def prepare_client(
    spec: ClientSpec,
    state_dir: Path,
    mcp_command: list[str],
    server_name: str = "drivingbench_sandbox",
    *,
    auth_home: Path | None = None,
    profile: str = "restricted",
    consolidate_driving_aliases: bool = False,
) -> PreparedClient:
    """Prepare neutral client state with restricted or ordinary CLI tools.

    State contains credentials and must live in ignored private artifacts, never published.
    The working directory is intentionally empty and outside the source checkout.
    """
    if profile not in {"restricted", "normal"}:
        raise ValueError("Unknown client tool profile")
    if consolidate_driving_aliases and profile != "normal":
        raise ValueError("Alias consolidation requires a normal private profile")
    if profile == "normal":
        _private_command(mcp_command)
    if not mcp_command or not Path(mcp_command[0]).is_absolute():
        raise ValueError("MCP executable must be absolute")
    if not server_name.replace("_", "").isalnum():
        raise ValueError("Invalid MCP server name")
    if not state_dir.exists():
        # Clients show the model paths under their config dir (Claude Code's memory path), so
        # the real state lives under a neutral name, linked from the run folder.
        real = STATE_ROOT / uuid.uuid4().hex
        real.mkdir(parents=True, mode=0o700)
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        state_dir.symlink_to(real)
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_dir.chmod(0o700)
    if profile == "restricted" and (state_dir / "client-profile.json").exists():
        raise ValueError("Cannot change a prepared normal profile to restricted")
    workspace_file = state_dir / "workspace.json"
    if workspace_file.exists():
        cwd = Path(json.loads(workspace_file.read_text()))
        cwd.mkdir(exist_ok=True)
    else:
        cwd = Path(tempfile.mkdtemp(prefix="driving-session-"))
        workspace_file.write_text(json.dumps(str(cwd)))
    # Drop parent Codex app/session pipes, plugin configuration and injected runtime state.
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("CODEX_", "CLAUDE_", "CURSOR_"))
    }
    server = {"command": mcp_command[0], "args": mcp_command[1:]}
    approved_tools = ["observe", "set_motion", "stop_now"] if is_offline_replay(mcp_command) else []
    if profile == "restricted":
        (state_dir / "mcp.json").write_text(json.dumps({"mcpServers": {server_name: server}}))
    if spec.family == "codex":
        config_dir = state_dir / "codex"
        config_dir.mkdir(exist_ok=True)
        source = (auth_home or Path.home() / ".codex") / "auth.json"
        if source.exists() and not (config_dir / "auth.json").exists():
            shutil.copyfile(source, config_dir / "auth.json")
            (config_dir / "auth.json").chmod(0o600)
        config = "\n".join(
            [
                'approval_policy = "never"',
                'sandbox_mode = "read-only"',
                'web_search = "disabled"',
                'cli_auth_credentials_store = "file"',
                "[features]",
                "shell_tool = false",
                f"[mcp_servers.{server_name}]",
                f"command = {json.dumps(server['command'])}",
                f"args = {json.dumps(server['args'])}",
            ]
        )
        if approved_tools:
            config += f"\nenabled_tools = {json.dumps(approved_tools)}\n"
            for tool in approved_tools:
                config += f'\n[mcp_servers.{server_name}.tools.{tool}]\napproval_mode = "approve"\n'
        if profile == "restricted":
            (config_dir / "config.toml").write_text(config + "\n")
        env["CODEX_HOME"] = str(config_dir)
    elif spec.family == "claude":
        config_dir = state_dir / "claude"
        config_dir.mkdir(exist_ok=True)
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        # Isolated config changes the keychain service name; use the existing OAuth
        # access token in memory only. Never serialize it into experiment artifacts.
        oauth = claude_oauth()
        # A copied token is never refreshed; every call would fail with a 401.
        if oauth.get("expiresAt", float("inf")) / 1000 < time.time() + 600:
            raise RuntimeError("Claude OAuth token expires soon; run `claude` to refresh")
        if oauth.get("accessToken"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth["accessToken"]
        source = (auth_home or Path.home() / ".claude") / ".credentials.json"
        if source.exists():
            shutil.copyfile(source, config_dir / ".credentials.json")
            (config_dir / ".credentials.json").chmod(0o600)
    elif spec.family == "cursor":
        # Cursor tells the model where its config lives (terminals and transcript folders in
        # the injected context), so that directory must not sit under the labeled trial path.
        pointer = state_dir / "cursor-state.json"
        if pointer.exists():
            config_dir = Path(json.loads(pointer.read_text()))
            config_dir.mkdir(exist_ok=True)
        else:
            config_dir = Path(tempfile.mkdtemp(prefix="driving-client-"))
            pointer.write_text(json.dumps(str(config_dir)))
        env["CURSOR_CONFIG_DIR"] = str(config_dir)
        env["CURSOR_DATA_DIR"] = str(config_dir)
        if profile == "restricted":
            (config_dir / "cli-config.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "permissions": {
                            "allow": [f"Mcp({server_name}:{tool})" for tool in approved_tools],
                            # The model drives with the MCP tools only: no shell, no files (the
                            # workspace mcp.json and the trial folder name the assignment), no web.
                            "deny": [
                                "Shell(*)",
                                "Read(**)",
                                "Read(/**)",
                                "Write(**)",
                                "Write(/**)",
                                "WebFetch(*)",
                            ],
                        },
                    }
                )
            )
        local = cwd / ".cursor"
        local.mkdir(exist_ok=True)
        global_mcp = Path.home() / ".cursor/mcp.json"
        inherited = (
            json.loads(global_mcp.read_text()).get("mcpServers", {}) if global_mcp.exists() else {}
        )
        # Cursor merges user and workspace MCP maps. Shadow EVERY inherited endpoint
        # with a harmless executable before invoking any Cursor command.
        isolated = {name: {"command": "/usr/bin/false", "args": []} for name in inherited}
        # The workspace mcp.json is greppable by the model, and the server's arguments name
        # the trial folder: register an opaque launcher and keep the real command out of reach.
        launcher = config_dir / f"{server_name}-mcp"
        launcher.write_text("#!/bin/sh\nexec " + shlex.join(mcp_command) + "\n")
        launcher.chmod(0o700)
        isolated[server_name] = {"command": str(launcher), "args": []}
        if profile == "restricted":
            (local / "mcp.json").write_text(json.dumps({"mcpServers": isolated}))
        for name in inherited if profile == "restricted" else ():
            if name != server_name:
                subprocess.run(
                    [spec.binary, "mcp", "disable", name],
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    check=True,
                    timeout=20,
                )
    client = PreparedClient(spec, state_dir, cwd, env, server_name, profile)
    if profile == "normal":
        source = (
            auth_home
            or Path.home()
            / {"codex": ".codex", "claude": ".claude", "cursor": ".cursor"}[spec.family]
        )
        _normal_profile(
            spec,
            state_dir,
            cwd,
            config_dir,
            source,
            mcp_command,
            server_name,
            consolidate=consolidate_driving_aliases,
        )
        assert_private_profile(client)
    return client
