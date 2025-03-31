# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.

import pydantic
import pydantic_settings


class RemoteAgentConfig(pydantic.BaseModel):
    """Configuration for a remote agent."""

    name: str = "default"
    base_url: pydantic.HttpUrl = pydantic.HttpUrl("http://0.0.0.0:3000")
    fetch_id_token: bool = False
    target_principal: str | None = None


class RemoteAgentConfigs(pydantic_settings.BaseSettings):
    """Configuration for multiple remote agents.

    Provide configuration through environment variables or CLI: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#usage
    """

    gemini: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            name="gemini",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/gemini"),
        )
    )
    guardrail: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            name="guardrail",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/gemini-with-guardrails"),
        )
    )
    function_calling: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            name="function_calling",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/function-calling"),
        )
    )
    semantic_router: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            name="semantic_router",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/semantic-router"),
        )
    )
    task_planner: RemoteAgentConfig = pydantic.Field(
        default_factory=lambda: RemoteAgentConfig(
            name="task_planner",
            base_url=pydantic.HttpUrl("http://0.0.0.0:3000/task-planner"),
        )
    )

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="demo_",
        case_sensitive=False,
        env_nested_delimiter="__",
        cli_parse_args=True,
    )
