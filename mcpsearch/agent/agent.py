import asyncio

from mcp.client.stdio import StdioServerParameters, stdio_client
from mlx_lm import generate, load
from rich import print

from .prompts import build_prompt


def load_llm():
    print("[blue]Loading LLM model...[/blue]")
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    print("[green]Model loaded.[/green]")
    return model, tokenizer


async def run_agent():

    model, tokenizer = load_llm()

    print("[blue]Connecting to MCP server...[/blue]")

    # THIS is the correct MCP configuration
    server = StdioServerParameters(
        command="python",
        args=["-m", "tools.book_tools_server"],
    )

    async with stdio_client(server) as streams:
        print("[green]Agent ready.[/green]")

        while True:
            question = input("\nAsk a question (or quit): ")

            if question.lower() == "quit":
                break

            prompt = build_prompt(question)

            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=300,
            )

            print("\n[bold green]Answer:[/bold green]\n")
            print(response)


if __name__ == "__main__":
    asyncio.run(run_agent())
