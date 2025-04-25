from openai import AsyncOpenAI
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from mcp import Tool
from utils.logtitles import log_title

@dataclass
class ToolCall:
    id: str
    function: Dict[str, Any]

class ChatOpenAI:
    def __init__(self, model:str='deepseek-chat', system_prompt:str='', tools:List[Tool] = [], context:str = ""):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.llm = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        self.tools = tools
        self.messages: List[AsyncOpenAI.Chat.ChatCompletionMessageParam] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        if context:
            self.messages.append({"role": "user", "content": context})

    async def chat(self, prompt:Optional[str]=None):
        log_title("Chat")
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        stream = await self.llm.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
            tools=self.getToolsDefinition(),
        )
        content = ""
        tool_calls: List[ToolCall] = []
        # 接下来不断读传出来的内容来解析
        log_title("Response")
        async for chunk in stream:
            delta = chunk.choices[0].delta
            # 处理content
            if delta.content:
                content_chunk = delta.content or "" # 如果content为空，则设置为空字符串
                content += content_chunk
                print(content_chunk, end="", flush=True)
            # 处理tool_calls
            if delta.tool_calls:
                for ToolCallChunk in delta.tool_calls:
                    # 第一次收到一个toolCall
                    if len(tool_calls) <= ToolCallChunk.index:
                        tool_calls.append(ToolCall(id = ToolCallChunk.id, function={'name':'', 'arguments':''}))
                    currentCall = tool_calls[ToolCallChunk.index]
                    if ToolCallChunk.id:
                        currentCall.id += ToolCallChunk.id
                    if ToolCallChunk.function.name:
                        currentCall.function['name'] += ToolCallChunk.function.name
                    if ToolCallChunk.function.arguments:
                        currentCall.function['arguments'] += ToolCallChunk.function.arguments
        self.messages.append(self.getAssistantMessage(content, tool_calls)) # 按照OpenAI文档包装toolcalls
        return content, tool_calls

    def getAssistantMessage(self, content:str, tool_calls:List[ToolCall]):
        """
        """
        message = {
            "role": "assistant",
            "content": content
        }
        if tool_calls:
            message["tool_calls"] = []
            for call in tool_calls:
                message["tool_calls"].append({
                    "type": "function",
                    "id": call.id,
                    "function":call.function
            })
        return message
    
    def AppendToolMessage(self, toolCallID:str, toolOutput:str):
        """
        添加工具调用返回的信息
        """
        self.messages.append({
            "role": "tool",
            "content": toolOutput,
            "tool_call_id": toolCallID
        })
    

    def getToolsDefinition(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        } for tool in self.tools]
