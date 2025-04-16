# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Search for documents in a LangGraph Store index."""

from typing import Callable, Awaitable

from langgraph.store import base
from concierge.tools import schemas
from google.genai import types as genai_types


def generate_langgraph_retriever(
    name: str,
    description: str,
    store: base.BaseStore,
    document_text_field: str,
    document_namespace: tuple[str],
) -> tuple[
    genai_types.FunctionDeclaration,
    Callable[[str, list[str]], Awaitable[schemas.DocumentSearchResult]],
]:
    retriever_fd = genai_types.FunctionDeclaration(
        response=None,
        description=description,
        name=name,
        parameters=genai_types.Schema(
            properties={
                "user_input": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Copy of the latest user input.",
                ),
                "search_queries": genai_types.Schema(
                    type=genai_types.Type.ARRAY,
                    items=genai_types.Schema(type=genai_types.Type.STRING),
                    description="A list of standalone search queries to perform semantic search against.",
                ),
            },
            required=["user_input", "search_queries"],
            property_ordering=["user_input", "search_queries"],
            type=genai_types.Type.OBJECT,
        ),
    )

    async def search(
        user_input: str,
        search_queries: list[str],
    ) -> schemas.InventorySearchResult:
        query = f"input: {user_input}\nqueries: " + ", ".join(search_queries)

        search_items = await store.asearch(document_namespace, query=query)

        documents = [
            schemas.RetrievedDocument(
                text=item.value[document_text_field],
                score=item.score,
            )
            for item in search_items
        ]

        return schemas.DocumentSearchResult(documents=documents)

    return retriever_fd, search
