# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Multi-turn RAG chat agent for the Concierge demo."""

import pydantic
from langgraph.store import base as store_base
from concierge import schemas, settings, utils
from concierge.langgraph_server import langgraph_agent
from concierge.nodes import chat, save_turn
from concierge.tools import retrieval

QNA_SYSTEM_PROMPT = """
You are a Q&A chat assistant that can answer questions about documents stored
in a vector search database. Always use your search tool provided to search
for documents before answering any query.

Not all retrieved documents may be relevant to the user query. If the content
of the documents is not enough to answer the user's question, explain that
you were unable to find enough information to provide an informed response.
""".strip()


class QNAConfig(pydantic.BaseModel):
    namespace: tuple[str, ...] = pydantic.Field(default_factory=tuple)


def load_agent(
    store: store_base.BaseStore,
    runtime_settings: settings.RuntimeSettings,
) -> langgraph_agent.LangGraphAgent:
    """Loads the function calling chat agent for the Concierge demo."""

    chat_node = chat.build_chat_node(
        node_name="chat",
        next_node="save-turn",
        system_prompt=QNA_SYSTEM_PROMPT,
        function_spec_loader=lambda config: load_function_specs(
            config=QNAConfig.model_validate(
                config.get("configurable", {}).get("qna_config", {})
            ),
            store=store,
            text_field=runtime_settings.retrieval_text_field,
        ),
    )

    save_turn_node = save_turn.build_save_turn_node(node_name="save-turn")

    gemini_agent = langgraph_agent.LangGraphAgent(
        state_graph=utils.load_graph(
            schema=chat.ChatState,
            nodes=[chat_node, save_turn_node],
            entry_point=chat_node,
        ),
        default_configurable={
            "chat_config": chat.ChatConfig(
                project=runtime_settings.project,
                region=runtime_settings.region,
                chat_model_name=runtime_settings.chat_model_name,
            ).model_dump(mode="json"),
        },
        checkpointer_config=runtime_settings.checkpointer,
    )

    return gemini_agent


def load_function_specs(
    config: QNAConfig,
    store: store_base.BaseStore,
    text_field: str,
) -> list[schemas.FunctionSpec]:
    """Load a list of function specs containing function declarations and callable handlers."""
    return [
        retrieval.generate_langgraph_retriever(
            name="search",
            description="Search for documents to ground the response.",
            store=store,
            document_namespace=config.namespace,
            document_text_field=text_field,
        )
    ]
