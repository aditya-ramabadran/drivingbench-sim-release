import json

from drivingbench.replay.tool_activity import read_activity, tool_activity


def test_provider_error_string_remains_an_infrastructure_failure(tmp_path):
    from drivingbench.replay.clients import ClientResult, result_errors

    message = "Selected model is at capacity. Please try a different model."
    events = [
        {"type": "error", "message": message},
        {"type": "turn.failed", "error": {"message": message}},
    ]
    path = tmp_path / "stdout.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = ClientResult(1, "session", path, tmp_path / "stderr")

    assert tool_activity(events) == []
    errors = result_errors(result, driving_servers={"drivingbench_sandbox"})
    assert errors[0] == "client_exit_1"
    assert len(errors) == 3
    assert all(message in error for error in errors[1:])


def claude(name, arguments, identity="call"):
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "id": identity, "name": name, "input": arguments}]
        },
    }


def test_codex_started_completed_and_failed_call_are_one_attempt():
    item = {
        "type": "mcp_tool_call",
        "id": "one",
        "server": "drivingbench_sandbox",
        "tool": "observe",
        "arguments": {},
    }
    events = [
        {"type": "item.started", "item": dict(item, status="in_progress")},
        {"type": "item.completed", "item": dict(item, status="failed")},
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "id": "two",
                "server": "codex_apps",
                "tool": "chatgpt_space.create_page_visualization",
                "status": "failed",
            },
        },
    ]
    rows = tool_activity(events)
    assert len(rows) == 2
    assert rows[0]["category"] == "driving" and rows[0]["status"] == "failed"
    assert rows[0]["event_indices"] == [1, 2]
    assert rows[1]["category"] == "off_task" and rows[1]["status"] == "failed"


def test_claude_targeted_discovery_and_unrelated_skill_read():
    events = [
        claude(
            "ToolSearch",
            {
                "query": "select:mcp__drivingbench_sandbox__observe,mcp__drivingbench_sandbox__stop_now"
            },
            "search",
        ),
        claude("mcp__drivingbench_sandbox__observe", {}, "drive"),
        claude("Read", {"path": "/plugins/x/skills/x-api-mcp-guide/SKILL.md"}, "read"),
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "read",
                        "is_error": True,
                        "content": "denied",
                    }
                ]
            },
        },
    ]
    rows = tool_activity(events)
    assert [row["category"] for row in rows] == ["discovery", "driving", "off_task"]
    assert rows[-1]["status"] == "failed" and rows[-1]["event_indices"] == [3, 4]


def test_cursor_old_dynamic_dispatch_is_not_a_read_or_shell_call():
    events = [
        claude("GetDynamicTools", {"namespace": "user-drivingbench_sandbox"}, "search"),
        claude(
            "CallDynamicTool",
            {
                "namespace": "user-drivingbench_sandbox",
                "toolName": "set_motion",
                "arguments": {"duration_s": 5},
            },
            "drive",
        ),
    ]
    rows = tool_activity(events)
    assert [row["category"] for row in rows] == ["discovery", "driving"]
    assert rows[-1]["server"] == "drivingbench_sandbox"
    assert rows[-1]["arguments"] == {"duration_s": 5}


def test_cursor_stream_and_custom_private_alias():
    events = []
    for subtype in ("started", "completed"):
        events.append(
            {
                "type": "tool_call",
                "subtype": subtype,
                "call_id": "one",
                "tool_call": {"getMcpToolsToolCall": {"args": {"server": "scene"}}},
            }
        )
        events.append(
            {
                "type": "tool_call",
                "subtype": subtype,
                "call_id": "two",
                "tool_call": {
                    "mcpToolCall": {
                        "args": {
                            "serverIdentifier": "scene",
                            "name": "scene-observe",
                            "toolName": "observe",
                            "args": {},
                        }
                    }
                },
            }
        )
    rows = tool_activity(events, driving_servers={"scene"})
    assert len(rows) == 2 and [row["category"] for row in rows] == ["discovery", "driving"]
    assert all(row["status"] == "completed" for row in rows)


def test_discovery_does_not_whitelist_other_targets_or_substring_spoofs():
    rows = tool_activity(
        [
            claude(
                "ToolSearch",
                {"query": "select:mcp__drivingbench_sandbox__observe,mcp__mail__send"},
                "a",
            ),
            claude("ToolSearch", {"query": "drivingbench and email tools"}, "b"),
            claude("GetDynamicTools", {"namespace": "user-not_drivingbench_sandbox"}, "c"),
            claude("GetDynamicTools", {"namespace": "user-drivingbench_sandbox"}, "d"),
        ]
    )
    assert [row["category"] for row in rows] == ["off_task", "unresolved", "off_task", "discovery"]


def test_codex_exec_wrapper_tracks_direct_tools_without_counting_quoted_examples():
    code = 'const sample = "tools.exec_command({cmd: nope})"; text(await tools.mcp__drivingbench_sandbox__observe({}));'
    event = {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "call_id": "wrapper",
            "name": "exec",
            "input": code,
        },
    }
    rows = tool_activity([event])
    assert len(rows) == 1 and rows[0]["category"] == "driving"
    assert rows[0]["wrapper"] == "exec" and rows[0]["tool"] == "observe"
    event["payload"]["input"] = (
        'text(await tools.mcp__drivingbench_sandbox__observe({})); await tools.exec_command({cmd:"ls"});'
    )
    assert [row["category"] for row in tool_activity([event])] == ["driving", "off_task"]


def test_arbitrary_computed_or_template_wrapper_code_stays_unresolved():
    for code in [
        "await tools[name]({})",
        'await fetch("https://example.com")',
        "`value ${tools.exec_command({})}`",
        'const x = "tools.mcp__drivingbench_sandbox__observe({})";',
    ]:
        event = {
            "payload": {"type": "custom_tool_call", "call_id": "one", "name": "exec", "input": code}
        }
        assert any(row["category"] == "unresolved" for row in tool_activity([event]))


def test_unknown_stream_call_remains_visible_and_partial_lines_keep_locations(tmp_path):
    path = tmp_path / "stdout.jsonl"
    path.write_text(
        "not-json\n"
        + json.dumps(
            {
                "type": "tool_call",
                "call_id": "opaque",
                "tool_call": {"futureFormat": {"value": "unknown"}},
            }
        )
        + '\n{"partial":'
    )
    rows = read_activity(path)
    assert len(rows) == 1 and rows[0]["category"] == "unresolved"
    assert rows[0]["event_indices"] == [2] and rows[0]["source"] == str(path)


def test_calls_without_ids_are_not_merged_and_text_is_not_a_tool_call():
    events = [
        claude("Read", {"path": "/x"}, None),
        claude("Read", {"path": "/x"}, None),
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "I will use Read."}]},
        },
    ]
    assert len(tool_activity(events)) == 2


def test_codex_native_tools_are_off_task():
    rows = tool_activity(
        [
            {"type": "item.completed", "item": {"id": str(i), "type": kind}}
            for i, kind in enumerate(("command_execution", "web_search", "file_change"))
        ]
    )
    assert len(rows) == 3 and all(row["category"] == "off_task" for row in rows)


def test_discovery_queries_and_older_cursor_target_names():
    events = [
        claude(
            "tool_search",
            {
                "queries": [
                    "+drivingbench_sandbox observe",
                    "drivingbench_sandbox set_motion stop_now tools",
                ]
            },
            "discovery",
        ),
        {
            "type": "tool_call",
            "call_id": "older",
            "tool_call": {
                "mcpToolCall": {"args": {"name": "drivingbench_sandbox-observe", "args": {}}}
            },
        },
        {"type": "item.completed", "item": {"type": "future_tool_call", "id": "new"}},
    ]
    assert [r["category"] for r in tool_activity(events)] == ["discovery", "driving", "unresolved"]


def test_multiline_quoted_template_is_not_an_executed_nested_tool():
    code = (
        "`example:\n tools.exec_command({})\n`; await tools.mcp__drivingbench_sandbox__observe({});"
    )
    event = {"payload": {"type": "custom_tool_call", "call_id": "a", "name": "exec", "input": code}}
    rows = tool_activity([event])
    assert len(rows) == 1 and rows[0]["category"] == "driving"


def test_completed_codex_arguments_replace_empty_start_arguments():
    item = {
        "id": "a",
        "type": "mcp_tool_call",
        "server": "drivingbench_sandbox",
        "tool": "set_motion",
    }
    rows = tool_activity(
        [{"item": item}, {"item": dict(item, arguments={"duration_s": 5}, status="completed")}]
    )
    assert len(rows) == 1 and rows[0]["arguments"] == {"duration_s": 5}


def test_cursor_result_only_completion_preserves_start_target_and_arguments():
    started = {
        "type": "tool_call",
        "subtype": "started",
        "call_id": "same-call",
        "tool_call": {
            "mcpToolCall": {
                "args": {"serverIdentifier": "scene", "toolName": "observe", "args": {"test": 1}}
            }
        },
    }
    completed = {
        "type": "tool_call",
        "subtype": "completed",
        "call_id": "same-call",
        "tool_call": {
            "mcpToolCall": {"result": {"rejected": {"reason": "User rejected MCP: scene-observe"}}}
        },
    }
    rows = tool_activity([started, completed], driving_servers={"scene"})
    assert len(rows) == 1
    assert rows[0]["category"] == "driving"
    assert rows[0]["server"] == "scene" and rows[0]["tool"] == "observe"
    assert rows[0]["arguments"] == {"test": 1}
    assert rows[0]["status"] == "failed" and rows[0]["event_indices"] == [1, 2]
    # No invented target when a completion has no matching start.
    assert tool_activity([completed], driving_servers={"scene"})[0]["category"] == "unresolved"


def test_cursor_result_only_completion_does_not_erase_discovery_intent():
    events = [
        {
            "type": "tool_call",
            "subtype": "started",
            "call_id": "search",
            "tool_call": {"getMcpToolsToolCall": {"args": {"server": "scene"}}},
        },
        {
            "type": "tool_call",
            "subtype": "completed",
            "call_id": "search",
            "tool_call": {"getMcpToolsToolCall": {"result": {"success": {}}}},
        },
    ]
    rows = tool_activity(events, driving_servers={"scene"})
    assert len(rows) == 1 and rows[0]["category"] == "discovery"
    assert rows[0]["arguments"] == {"server": "scene"} and rows[0]["status"] == "completed"
