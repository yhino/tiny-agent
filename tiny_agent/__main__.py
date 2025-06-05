import json
from datetime import datetime
from typing import Generator, Any
from zoneinfo import ZoneInfo

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()


def current_time(timezone: str = "Asia/Tokyo") -> str:
    """Returns the current time in ISO 8601 format"""
    tz = ZoneInfo(timezone)
    return datetime.now().astimezone(tz).isoformat()


class TinyAgent:
    def __init__(self, llm_client: Anthropic):
        self.llm_client = llm_client
        self.tools = [
            {
                "name": "current_time",
                "description": "Returns the current time in ISO 8601 format",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Specifies the time zone in which to return the current time",
                            "default": "Asia/Tokyo",
                        },
                    },
                    "required": [],
                },
            }
        ]
        self.mapping_tool_function = {
            "current_time": current_time,
        }

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        result = self.mapping_tool_function[tool_name](**tool_args)
        if result is None:
            result = "The operation completed but didn't return any results."
        elif isinstance(result, list):
            result = ", ".join(result)
        elif isinstance(result, dict):
            result = json.dumps(result)
        else:
            result = str(result)
        return result

    def run_interactive_session(self) -> None:
        while True:
            try:
                console.print()
                query = Prompt.ask("[green]Query[/green]").strip()
                if query.lower() == "quit":
                    break

                console.print()
                for response in self.process_query(query):
                    console.print(Markdown(response))
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

    def shutdown(self):
        pass

    def process_query(self, query: str) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": query}]
        response = self.llm_client.messages.create(
            max_tokens=1024,
            model="claude-3-5-sonnet-latest",
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

                    result = self.execute_tool(tool_name, tool_args)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result,
                                }
                            ],
                        }
                    )
                    response = self.llm_client.messages.create(
                        max_tokens=1024,
                        model="claude-3-5-sonnet-latest",
                        tools=self.tools,
                        messages=messages,
                    )
                    if len(response.content) == 1 and response.content[0] == "text":
                        yield response.content[0].text
                        process_query = False


def main() -> None:
    load_dotenv()
    llm_client = Anthropic()
    agent = TinyAgent(llm_client)

    try:
        agent.run_interactive_session()
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
