from typing import Generator

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()


class TinyAgent:
    def __init__(self, llm_client: Anthropic):
        self.llm_client = llm_client

    def run_interactive_session(self) -> None:
        while True:
            try:
                query = Prompt.ask("[green]Query[/green]").strip()
                if query.lower() == "quit":
                    break
                console.print()
                for response in self.process_query(query):
                    console.print(Markdown(response))
                console.print()
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")

    def shutdown(self):
        pass

    def process_query(self, query: str) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": query}]
        response = self.llm_client.messages.create(
            max_tokens=1024,
            model="claude-3-5-sonnet-latest",
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
