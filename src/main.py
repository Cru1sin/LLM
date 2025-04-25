import asyncio
from ChatOpenAI import ChatOpenAI
from mcp_client import MCPClient
from agent import Agent
from EmbeddingRetrivers import EmbeddingRetriever
from utils.logtitles import log_title
import os

async def main():
    # 获取 LLM 根目录路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    fetchmcp = MCPClient('fetch', 'uvx', ['mcp-server-fetch'])
    filemcp = MCPClient('file', 'npx', ['-y', '@modelcontextprotocol/server-filesystem', project_root])

    embedding_retriever = EmbeddingRetriever(embedding_model='BAAI/bge-m3')
    await embedding_retriever.embed_konwledge(os.path.join(project_root, 'knowledge'))

    query = "根据Bret的信息，创建一个Ta的故事，保存到{project_root}/output/Bret.md文件中, 要包含TA的基本信息和故事"
    context = await embedding_retriever.retrieve_context(query)
    # 使用 async with 来自动管理 Agent 的初始化和清理
    async with Agent(model='deepseek-chat', llm=ChatOpenAI(), mcpClients=[fetchmcp, filemcp], context=context) as agent:
        # 调用 invoke 函数
        response = await agent.invoke(query)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())