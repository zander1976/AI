"""
Prompt definitions for the book research agent.

Keeping prompts in a separate file makes them easier to modify
without touching the agent logic.
"""

SYSTEM_PROMPT = """
You are an AI research assistant with access to a library of books.

You can use tools to search the library.

Available tools:
- search_library
- get_best_quote
- list_books
- summarize_book

Guidelines:
- Use tools when you need information from books
- Cite the book name when quoting text
- If the user asks what books exist, call list_books
- If the user asks for a quote, use get_best_quote
- If the user asks a general question about a topic, use search_library
"""


def build_prompt(user_question):
    """
    Construct the full prompt for the LLM.
    """

    prompt = SYSTEM_PROMPT + "\n\n"

    prompt += "User: " + user_question + "\n"
    prompt += "Assistant:"

    return prompt
