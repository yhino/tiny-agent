import asyncio
from contextlib import AsyncExitStack
from typing import AsyncGenerator

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, stdio_client, StdioServerParameters
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()


class Config(BaseSettings):
    max_tokens: int = 2024
    model_name: str = "claude-sonnet-4-20250514"


class TinyAgent:
    def __init__(self, config: Config | None):
        self.config = config or Config()
        self.llm_client = Anthropic()
        self.session: ClientSession | None = None
        self.tools: list[dict] = []
        self.exit_stack = AsyncExitStack()

    async def connect_mcp_servers(self) -> None:
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "servers/clock.py"],
            env=None,
        )
        read, write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.session = session

        response = await session.list_tools()
        self.tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
        console.print(
            "Connected to server with tools:",
            [tool.name for tool in response.tools],
        )

    async def run_interactive_session(self) -> None:
        while True:
            try:
                console.print()
                query = Prompt.ask("[green]Query[/green]").strip()
                if query.lower() == "quit":
                    break

                console.print()
                async for response in self.process_query(query):
                    console.print(Markdown(response))
            except Exception as exc:
                console.print(f"[red]Error: {str(exc)}[/red]")

    async def shutdown(self):
        await self.exit_stack.aclose()

    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        messages = [{"role": "user", "content": query}]
        response = self.llm_client.messages.create(
            max_tokens=self.config.max_tokens,
            model=self.config.model_name,
            tools=self.tools,
            messages=messages,
        )
        process_query = True
        while process_query:
            assistant_content = []
            for content in response.content:
                if content.type == "text":
                    yield content.text
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        process_query = False
                if content.type == "tool_use":
                    assistant_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_content})

                    tool_id, tool_args, tool_name = (
                        content.id,
                        content.input,
                        content.name,
                    )
                    yield f"Calling tool {tool_name} with args {tool_args}"

                    result = await self.session.call_tool(
                        tool_name,
                        arguments=tool_args,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result.content,
                                }
                            ],
                        }
                    )
                    response = self.llm_client.messages.create(
                        max_tokens=self.config.max_tokens,
                        model=self.config.model_name,
                        tools=self.tools,
                        messages=messages,
                    )
                    if len(response.content) == 1 and response.content[0] == "text":
                        yield response.content[0].text
                        process_query = False


async def main() -> None:
    load_dotenv()
    agent = TinyAgent(Config())

    try:
        await agent.connect_mcp_servers()
        await agent.run_interactive_session()
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
