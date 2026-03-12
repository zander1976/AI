#!/usr/bin/env python3

"""
BOOK MCP TOOL SERVER

This file exposes tools that the AI agent can call.

The tools provide access to the book retrieval system.

Architecture:

agent
 ↓
MCP tools (this file)
 ↓
retrieval system
 ↓
vector index
"""

from mcp.server.fastmcp import FastMCP

from retrieval import VectorSearch

# create MCP server
mcp = FastMCP("book-library")


# create search engine
search_engine = VectorSearch()


@mcp.tool()
def search_library(query: str) -> str:
    """
    Search across all books for relevant passages.
    """

    results = search_engine.search(query, k=5)

    output = ""

    for r in results:
        output += f"Book: {r['book']}\n"

        output += r["text"]

        output += "\n\n"

    return output


@mcp.tool()
def get_best_quote(query: str) -> str:
    """
    Return the single best quote related to a topic.
    """

    result = search_engine.search(query, k=1)[0]

    quote = result["text"]

    book = result["book"]

    return f"{quote}\n\n(Source: {book})"


@mcp.tool()
def list_books() -> str:
    """
    List all books available in the library.
    """

    books = set()

    for m in search_engine.metadata:
        books.add(m["book"])

    return "\n".join(sorted(books))


if __name__ == "__main__":
    print("Starting book MCP server...")

    mcp.run()
