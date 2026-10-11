"""Auditable tool activity from CLI streams and exported driving transcripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from drivingbench.replay.clients import DRIVING_SERVERS

DRIVING_TOOLS = {"observe", "set_motion", "stop_now"}
DISCOVERY = {"ToolSearch", "GetDynamicTools", "getMcpToolsToolCall", "search_tool", "tool_search"}
WRAPPERS = {"exec", "functions.exec", "multi_tool_use.parallel"}


def _arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value if value is not None else {}


def _server(value):
    return value.removeprefix("user-") if isinstance(value, str) else ""


def _target(name, server=""):
    if name.startswith("mcp__"):
        pieces = name.split("__", 2)
        if len(pieces) == 3:
            return pieces[1], pieces[2]
    if name.startswith("mcp."):
        pieces = name.split(".", 2)
        if len(pieces) == 3:
            return pieces[1], pieces[2]
    return _server(server), name


def _discovery(args, driving):
    if not isinstance(args, dict):
        return "unresolved"
    target = args.get("namespace") or args.get("server")
    if target:
        return "discovery" if _server(target) in driving else "off_task"
    if "queries" in args:
        queries = args["queries"]
        if not isinstance(queries, list) or not queries:
            return "unresolved"
        categories = [_discovery({"query": query}, driving) for query in queries]
        return (
            "off_task"
            if "off_task" in categories
            else "discovery"
            if set(categories) == {"discovery"}
            else "unresolved"
        )
    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return "unresolved"
    if query.startswith("select:"):
        targets = re.split(r"[,\s]+", query.removeprefix("select:").strip())
        return "discovery" if all(_target(name)[0] in driving for name in targets) else "off_task"
    if _server(query.strip()) in driving:
        return "discovery"
    words = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", query))
    if words & driving and words <= driving | DRIVING_TOOLS | {
        "mcp",
        "tools",
        "tool",
        "find",
        "search",
    }:
        return "discovery"
    # Free-text search intent is ambiguous; a matching substring is not sufficient.
    return "unresolved"


def _classify(name, args, server, driving):
    if name == "CallDynamicTool":
        if not isinstance(args, dict) or not args.get("namespace") or not args.get("toolName"):
            return "unresolved", server, name, args
        server, name, args = _server(args["namespace"]), args["toolName"], args.get("arguments", {})
    if name in DISCOVERY:
        return _discovery(args, driving), server, name, args
    server, name = _target(name, server)
    if not server:
        for alias in driving:
            if name.startswith(alias + "-") and name[len(alias) + 1 :] in DRIVING_TOOLS:
                server, name = alias, name[len(alias) + 1 :]
                break
    if server:
        category = "driving" if server in driving and name in DRIVING_TOOLS else "off_task"
    else:
        category = "off_task" if name and name not in DRIVING_TOOLS else "unresolved"
    return category, server, name, args


def _calls(event):
    """Yield call starts/completions; tool results only update an already observed call."""
    item = event.get("item", {})
    kind = item.get("type")
    if kind == "mcp_tool_call":
        yield (
            item.get("id"),
            item.get("tool", ""),
            item.get("arguments"),
            item.get("server", ""),
            item.get("status", event.get("type")),
            False,
        )
    elif kind in {"command_execution", "web_search", "file_change"}:
        yield (
            item.get("id"),
            kind,
            item.get("command", item.get("query", {})),
            "",
            item.get("status", event.get("type")),
            False,
        )
    elif kind in {"function_call", "custom_tool_call", "tool_call"}:
        yield (
            item.get("call_id", item.get("id")),
            item.get("name", ""),
            item.get("arguments", item.get("input")),
            "",
            event.get("type"),
            False,
        )
    elif isinstance(kind, str) and kind.endswith("tool_call"):
        yield item.get("id"), "", {}, "", event.get("type"), False
    payload = event.get("payload", {})
    if payload.get("type") in {"function_call", "custom_tool_call"}:
        yield (
            payload.get("call_id"),
            payload.get("name", ""),
            payload.get("arguments", payload.get("input")),
            "",
            "attempted",
            False,
        )
    if event.get("type") == "tool_call":
        outer = event.get("tool_call", {})
        found = False
        for name, call in outer.items():
            if not name.endswith("ToolCall") or not isinstance(call, dict):
                continue
            found = True
            args = call.get("args", {})
            server = ""
            if name == "mcpToolCall":
                server = args.get("serverIdentifier") or args.get("providerIdentifier", "")
                name = args.get("toolName", args.get("name", ""))
                if server and name.startswith(server + "-"):
                    name = name[len(server) + 1 :]
                args = args.get("args", {})
            result = call.get("result", {})
            failed = isinstance(result, dict) and any(
                k in result for k in ("error", "rejected", "permissionDenied")
            )
            yield (
                event.get("call_id", outer.get("toolCallId")),
                name,
                args if "args" in call else None,
                server,
                "failed" if failed else event.get("subtype", "attempted"),
                False,
            )
        if not found:
            yield event.get("call_id"), "", {}, "", event.get("subtype", "attempted"), False
    message = event.get("message", {})
    content = message.get("content", []) if isinstance(message, dict) else []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"tool_use", "server_tool_use"}:
                yield (
                    block.get("id"),
                    block.get("name", ""),
                    block.get("input"),
                    "",
                    "attempted",
                    False,
                )
            elif block.get("type") == "tool_result":
                yield (
                    block.get("tool_use_id"),
                    "",
                    {},
                    "",
                    "failed" if block.get("is_error") else "completed",
                    True,
                )


def _nested(code):
    """Static direct tool targets only; arbitrary/computed JavaScript remains unresolved."""
    if not isinstance(code, str):
        return [], True
    # Mask strings/comments before matching: quoted examples are not executed calls.
    masked = re.sub(r"(['\"`])(?:\\[\s\S]|(?!\1)[\s\S])*\1|//[^\n]*|/\*[\s\S]*?\*/", "''", code)
    targets = re.findall(r"\btools\.([A-Za-z_][A-Za-z_0-9]*)\s*\(", masked)
    rest = re.sub(r"\btools\.[A-Za-z_][A-Za-z_0-9]*\s*\(", "(", masked)
    rest = re.sub(r"\bawait\b", "", rest)
    unknown = bool(
        re.search(
            r"\btools\b|\b(?:fetch|require|import|eval)\b|[A-Za-z_$][\w.$]*\s*\(",
            re.sub(
                r"\b(?:text|image|audio|notify|Promise\.all|Promise\.allSettled)\s*\(", "(", rest
            ),
        )
    )
    return targets, unknown or not targets or "${" in code


def tool_activity(events, *, driving_servers=DRIVING_SERVERS, source=None):
    """Classify attempts, including denied calls; unresolved records require review."""
    driving = set(driving_servers)
    rows = {}
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            continue
        for ordinal, (identity, name, raw, server, status, result_only) in enumerate(_calls(event)):
            key = str(identity) if identity else f"event:{index}:{ordinal}"
            if result_only:
                for row in rows.values():
                    if row["call_id"] == key:
                        row["status"] = status
                        row["event_indices"].append(index)
                continue
            args = _arguments(raw)
            candidates = [(name, args, server, False)]
            if name in WRAPPERS:
                targets, unknown = _nested(
                    args
                    if isinstance(args, str)
                    else args.get("code")
                    if isinstance(args, dict)
                    else None
                )
                candidates = [(target, {}, "", False) for target in targets]
                if unknown:
                    candidates.append((name, args, "", True))
            for part, (target, arguments, host, unknown) in enumerate(candidates):
                category, host, target, arguments = _classify(target, arguments, host, driving)
                row_key = (key, part)
                if row_key in rows:
                    row = rows[row_key]
                    row["status"] = status
                    row["event_indices"].append(index)
                    if raw is not None:
                        row.update(
                            tool=target,
                            server=host or None,
                            arguments=arguments,
                            category="unresolved" if unknown else category,
                        )
                    continue
                rows[row_key] = {
                    "call_id": key,
                    "event_indices": [index],
                    "source": source,
                    "tool": target,
                    "server": host or None,
                    "arguments": arguments,
                    "category": "unresolved" if unknown else category,
                    "status": status or "attempted",
                    "wrapper": name if name in WRAPPERS else None,
                }
    return list(rows.values())


def read_activity(path, *, driving_servers=DRIVING_SERVERS):
    """Retain physical source line numbers even if a stream has partial/non-JSON lines."""
    path = Path(path)
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            events.append(None)
    return tool_activity(events, driving_servers=driving_servers, source=str(path))
