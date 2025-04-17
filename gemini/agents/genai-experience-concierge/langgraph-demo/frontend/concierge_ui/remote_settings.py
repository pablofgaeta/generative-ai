# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Settings to configure connection to remote agents."""

import pydantic
import pydantic_settings


class RemoteConfig(pydantic.BaseModel):
    base_url: pydantic.HttpUrl = pydantic.HttpUrl("http://0.0.0.0:3000")
    fetch_id_token: bool = False
    target_principal: str | None = None


class StoreConfig(RemoteConfig):
    retrieval_text_field: str


class RemoteAgentConfig(RemoteConfig):
    """Configuration for a remote agent."""

    agent_id: str = "default"


class RemoteAgentConfigs(pydantic_settings.BaseSettings):
    """Configuration for multiple remote agents.

    Provide configuration through environment variables or CLI: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#usage
    """  # pylint: disable=line-too-long

    store: StoreConfig = pydantic.Field(
        default_factory=lambda: StoreConfig(
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/default_store"),
            retrieval_text_field="text",
        )
    )

    gemini: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            agent_id="gemini",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/gemini"),
        )
    )
    guardrail: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            agent_id="guardrails",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/gemini-with-guardrails"),
        )
    )
    function_calling: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            agent_id="function-calling",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/function-calling"),
        )
    )
    semantic_router: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            agent_id="semantic-router",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/semantic-router"),
        )
    )
    task_planner: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            agent_id="task-planner",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/task-planner"),
        )
    )
    qna: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            agent_id="document-qna",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/qna"),
        )
    )
    qna_store_namespace: tuple[str, ...] = ("qna",)

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="demo_",
        case_sensitive=False,
        env_nested_delimiter="__",
        cli_parse_args=True,
    )
