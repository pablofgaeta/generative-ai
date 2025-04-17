# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Common schemas for the concierge demo."""

from typing import Callable, NamedTuple, Protocol, TypedDict

from langchain_core.runnables import config as lc_config
from google.genai import types as genai_types
import pydantic


class Node[T, **P](NamedTuple):
    """
    Represents a node in a LangGraph.

    Attributes:
        name: The name of the node.
        action: The action associated with the node.
    """

    name: str
    """The name of the node."""

    fn: Callable[P, T]
    """The action associated with the node."""


class Coordinate(pydantic.BaseModel):
    """Represents a coordinate with its latitude and longitude."""

    latitude: float = pydantic.Field(ge=-90, le=90)
    """Latitude of the coordinate."""
    longitude: float = pydantic.Field(ge=-180, le=180)
    """Longitude of the coordinate."""


class BaseTurn(TypedDict, total=False):
    """Represents a single turn in a conversation.

    May be extended, but attributes cannot be overwritten.
    """

    user_input: str
    """The user's input for this turn."""

    response: str
    """The agent's response for this turn, if any."""

    messages: list[genai_types.Content]
    """List of Gemini Content objects representing the conversation messages in this turn."""


class FunctionSpec(NamedTuple):
    """A named tuple representing an LLM-callable function specification."""

    fd: genai_types.FunctionDeclaration
    callable: Callable


class RuntimeFunctionSpecLoader(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for a function specification loader."""

    def __call__(self, config: lc_config.RunnableConfig) -> list[FunctionSpec]:
        """Load a list of function specs dynamically at runtime based on the runtime config."""
