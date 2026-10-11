"""Private process-local wording variants around the unchanged three-tool MCP."""

import argparse
import asyncio
from urllib.parse import urlsplit

import httpx
from drivingbench.mcp.tools import make_server
from drivingbench.replay.protocol import VARIANTS


def study_server(client, variant="reference", label="MCP"):
    config = VARIANTS[variant]
    server = make_server(client, label)
    server._mcp_server.name = config["server_name"]
    server._mcp_server.instructions = server._mcp_server.instructions.replace(
        "drivingbench_sandbox", config["server_name"]
    )
    if config["tool_suffix"]:
        for tool in server._tool_manager.list_tools():
            tool.description += "\n\n" + config["tool_suffix"]
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="reference")
    parser.add_argument("--client", default="MCP")
    args = parser.parse_args()
    url = urlsplit(args.gateway_url)
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.port in (8766, 8876, 8877):
        parser.error("study gateway must be a dedicated 127.0.0.1 HTTP port")

    async def run():
        async with httpx.AsyncClient(
            base_url=args.gateway_url, timeout=60, trust_env=False
        ) as client:
            await study_server(client, args.variant, args.client).run_stdio_async()

    asyncio.run(run())


if __name__ == "__main__":
    main()
