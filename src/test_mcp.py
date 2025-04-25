from ChatOpenAI import ChatOpenAI
from mcp_client import MCPClient
import asyncio

async def main():
    fetchmcp = MCPClient('fetch', 'uvx', ['mcp-server-fetch'])
    await fetchmcp.init()
    tools = await fetchmcp.getTools()
    print(tools)
    await fetchmcp.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
    