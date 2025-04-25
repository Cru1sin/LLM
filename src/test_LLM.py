from ChatOpenAI import ChatOpenAI
import asyncio
import time

async def main():
    start = time.time()
    results = await asyncio.gather(
        ChatOpenAI().chat()
    )

    for i, (content, toolCalls) in enumerate(results):
        print(f"\nResult {i+1}: {content}")
        print(f"Tool Calls: {toolCalls}")

    print(f"\n耗时: {time.time() - start:.2f} 秒")

if __name__ == "__main__":
    asyncio.run(main())
    