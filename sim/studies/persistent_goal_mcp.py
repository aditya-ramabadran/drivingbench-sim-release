"""Private framing arm around the unchanged three-tool MCP."""

import argparse
import asyncio
import json
import sys
from urllib.parse import urlsplit

import httpx
from drivingbench.mcp.tools import make_server
from drivingbench.replay.fork import SERVER_NAME as REALVEHICLE_SERVER
from drivingbench.replay.fork import VEHICLE
from drivingbench.shared.contracts import SERVER_NAME
from drivingbench.sim.run import isolated_command


def study_server(client, *, realism=False, label="MCP"):
    server = make_server(client, label)
    if realism:
        server._mcp_server.name = REALVEHICLE_SERVER
        server._mcp_server.instructions = server._mcp_server.instructions.replace(
            SERVER_NAME, REALVEHICLE_SERVER
        )
        tool = server._tool_manager.get_tool("observe")
        original = tool.fn

        async def observe():
            content = await original()
            summary = json.loads(content[0].text)
            summary["vehicle"] = VEHICLE
            return [
                content[0].model_copy(update={"text": json.dumps(summary, separators=(",", ":"))}),
                *content[1:],
            ]

        tool.fn = observe
    return server


def configuration(row, gateway):
    from drivingbench.sim.studies.persistent_goal import ARMS

    if row["arm"] not in ARMS:
        raise ValueError("Unknown persistent-goal arm")
    if row["arm"] != "straight_realism":
        return isolated_command(gateway), SERVER_NAME
    return [
        sys.executable,
        "-m",
        "drivingbench.sim.studies.persistent_goal_mcp",
        "--gateway-url",
        gateway,
        "--realism",
    ], REALVEHICLE_SERVER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--realism", action="store_true")
    parser.add_argument("--client", default="MCP")
    args = parser.parse_args()
    try:
        url = urlsplit(args.gateway_url)
        dedicated = (
            url.scheme == "http"
            and url.hostname == "127.0.0.1"
            and url.port is not None
            and url.port not in (8766, 8876, 8877)
            and url.username is None
            and url.password is None
        )
    except ValueError:
        dedicated = False
    if not dedicated:
        parser.error("study gateway must be a dedicated 127.0.0.1 HTTP port")

    async def run():
        async with httpx.AsyncClient(
            base_url=args.gateway_url, timeout=60, trust_env=False
        ) as client:
            await study_server(client, realism=args.realism, label=args.client).run_stdio_async()

    asyncio.run(run())


if __name__ == "__main__":
    main()
