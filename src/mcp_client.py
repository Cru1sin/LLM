import asyncio
from typing import Optional, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self, name:str, command:str, args:List[str]):
        # Initialize session and client objects
        self.tools = List[Tool]
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.name = name
        self.command = command
        self.args = args

    async def __aenter__(self):
        await self.exit_stack.__aenter__()      # 进入 exit stack 上下文管理器
        await self.connect_to_server()          # 启动 MCP 服务器连接
        return self                             # 返回实例本身，赋给 as 后面的变量

    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.aclose()          # 自动关闭所有在 exit_stack 注册的上下文对象（如 client session）
    
    async def getTools(self):
        return self.tools

    async def connect_to_server(self):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=None
        ) # 服务器参数
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    def get_tools(self):
        return self.tools
    
    def call_tool(self, tool_name:str, arguments:str):
        return self.session.call_tool(tool_name, arguments)