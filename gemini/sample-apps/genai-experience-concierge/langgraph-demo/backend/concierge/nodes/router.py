# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Semantic router node for routing user queries to sub-agents."""

import enum
import logging
from typing import Generic, Literal, TypedDict, TypeVar

from concierge import schemas, utils
from google import genai
from google.genai import types as genai_types
from langchain_core.runnables import config as lc_config
from langgraph import types as lg_types
from langgraph.config import get_stream_writer
import pydantic

logger = logging.getLogger(__name__)

RouterTarget = TypeVar("RouterTarget", bound=enum.Enum)


class RouterClassification(pydantic.BaseModel, Generic[RouterTarget]):
    """Structured classification output for routing user queries."""

    reason: str = pydantic.Field(
        description="Reason for classifying the latest user query."
    )
    """Explanation of why the query was classified to a specific target."""

    target: RouterTarget
    """The target node to route the query to."""

    model_config = pydantic.ConfigDict(
        json_schema_extra={"propertyOrdering": ["reason", "target"]}
    )
    """Configuration to specify the ordering of properties in the JSON schema."""


class RouterTurn(schemas.BaseTurn, Generic[RouterTarget]):
    """Represents a single turn in a conversation."""

    router_classification: RouterClassification[RouterTarget] | None
    """The router classification for the current turn."""


class RouterState(TypedDict, Generic[RouterTarget], total=False):
    """Stores the active turn and conversation history."""

    current_turn: RouterTurn[RouterTarget] | None
    """The current conversation turn."""

    turns: list[RouterTurn[RouterTarget]]
    """List of all conversation turns in the session."""


class RouterConfig(pydantic.BaseModel):
    """Configuration settings for the router node."""

    project: str
    """The Google Cloud project ID."""
    region: str
    """The Google Cloud region."""
    router_model_name: str
    """The name of the Gemini intent detection model."""
    max_router_turn_history: int
    """The maximum number of prior turns to include in the conversation history."""


def build_semantic_router_node(
    node_name: str,
    system_prompt: str,
    target_nodes: dict[RouterTarget, str],
) -> schemas.Node:
    """
    Builds a LangGraph node that can dynamically route between sub-agents based on user intent.
    """

    # ignore typing errors, this creates a valid literal type
    NextNodeT = Literal[*target_nodes]  # type: ignore

    async def ainvoke(
        state: RouterState[RouterTarget],
        config: lc_config.RunnableConfig,
    ) -> lg_types.Command[NextNodeT]:
        """
        Asynchronously invokes the router node to classify user input and determine the next action.

        This function takes the current conversation state and configuration, interacts with the
        Gemini model to classify the user's input based on predefined categories, and
        determines which sub-agent should handle the request.

        Runtime configuration should be passed in `config.configurable.router_config`.

        Args:
            state: The current state of the conversation session, including user input and history.
            config: The LangChain RunnableConfig containing agent-specific configurations.

        Returns:
            A Command object that specifies the next node to transition to
            and the updated conversation state. This state includes the router classification.
        """

        router_config = RouterConfig.model_validate(
            config.get("configurable", {}).get("router_config", {})
        )

        stream_writer = get_stream_writer()

        current_turn = state.get("current_turn")
        assert current_turn is not None, "current turn must be set"

        # Initialize generate model
        client = genai.Client(
            vertexai=True,
            project=router_config.project,
            location=router_config.region,
        )

        user_content = utils.load_user_content(current_turn=current_turn)
        contents = [
            content
            for turn in state.get("turns", [])[-router_config.max_router_turn_history :]
            for content in turn.get("messages", [])
        ] + [user_content]

        # generate streaming response
        response = await client.aio.models.generate_content(
            model=router_config.router_model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                candidate_count=1,
                temperature=0.2,
                seed=0,
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=RouterClassification[RouterTarget],
            ),
        )

        router_classification = RouterClassification[RouterTarget].model_validate_json(
            response.text or ""
        )

        stream_writer(
            {
                "router_classification": {
                    "target": router_classification.target.value,
                    "reason": router_classification.reason,
                }
            }
        )

        current_turn["router_classification"] = router_classification

        next_node = None
        for target_value, target_node in target_nodes.items():
            if router_classification.target == target_value:
                next_node = target_node
                break
        else:
            raise RuntimeError(
                f"Unhandled router classification target: {router_classification.target}"
            )

        return lg_types.Command(
            update=RouterState(current_turn=current_turn),
            goto=next_node,
        )

    return schemas.Node(name=node_name, fn=ainvoke)
