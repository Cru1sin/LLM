from contextlib import AsyncExitStack
from mcp_client import MCPClient
from typing import List, Optional
from ChatOpenAI import ChatOpenAI
from utils.logtitles import log_title
import json
from mcp import Tool

class Agent:
    def __init__(self, model:str, llm:Optional[ChatOpenAI], mcpClients: List[MCPClient] = [], system_prompt:str = '', context:str = '', tools:List[Tool] = []):
        self.mcpClients: List[MCPClient] = mcpClients
        self.llm = llm
        self.model = model
        self.context = context
        self.system_prompt = system_prompt
        self.tools = tools
        self.exit_stack = AsyncExitStack()  # 用于管理异步上下文

    async def __aenter__(self):
        # 在 async with 进入时调用 init() 来初始化
        log_title("Initializing MCP Clients")
        await self.exit_stack.__aenter__()
        
        for mcpClient in self.mcpClients:
            await self.exit_stack.enter_async_context(mcpClient)  # 将 MCPClient 加入 exit_stack
            
            tools = await mcpClient.getTools()
            self.tools.extend(tools)

        self.llm = ChatOpenAI(self.model, self.system_prompt, self.tools, self.context)  # 初始化 LLM
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # 在 async with 退出时自动清理资源
        log_title("Closing MCP Clients")
        await self.exit_stack.aclose()  # 自动清理所有注册的上下文对象

    async def invoke(self, prompt:str):
        if self.llm is None:
            raise ValueError("LLM is not initialized")
        
        content, tool_calls = await self.llm.chat(prompt)
        while True:
            if tool_calls:
                for tool_call in tool_calls:
                    mcpClient = None
                    for mcp in self.mcpClients:
                        # 获取 MCPClient 的工具列表
                        tools = await mcp.getTools()  # 获取工具列表
                        if any(tool.name == tool_call.function['name'] for tool in tools):
                            mcpClient = mcp
                            break
                    if mcpClient:
                        log_title("TOOL USE " + f"{tool_call.function['name']}")
                        print(f"Calling Tool: {tool_call.function['name']}")
                        print(f"Arguments: {tool_call.function['arguments']}")
                        tool_result = await mcpClient.call_tool(tool_call.function['name'], json.loads(tool_call.function['arguments']))
                        print(f"Tool Result: {tool_result.content}")
                        self.llm.AppendToolMessage(tool_call.id, tool_result.content[0].text)
                    else:
                        self.llm.AppendToolMessage(tool_call.id, "MCP Client not found")
                content, tool_calls = await self.llm.chat()
                continue
            return content