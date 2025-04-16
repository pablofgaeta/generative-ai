# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""FastAPI server for hosting LangGraph agents."""

import contextlib
from typing import AsyncGenerator

from concierge import settings, utils
from concierge.agents import (
    function_calling,
    gemini,
    guardrails,
    semantic_router,
    task_planner,
    qna,
)
from google import genai
from concierge.langgraph_server import fastapi_app, store
import fastapi

# Build compiled LangGraph agents with optional checkpointer based on config

runtime_settings = settings.RuntimeSettings()

index_store = store.load_store(
    backend_config=runtime_settings.store,
    index={
        "dims": 768,
        "embed": utils.create_retriever(
            model=runtime_settings.embedding_model_name,
            batch_size=10,
            dimensionality=768,
            client=genai.Client(
                vertexai=True,
                project=runtime_settings.project,
                location=runtime_settings.region,
            ),
        ),
        "fields": [runtime_settings.retrieval_document_text_field],
    },
    ttl=(
        runtime_settings.retrieval_ttl.model_dump(mode="json")
        if runtime_settings.retrieval_ttl
        else None
    ),
)

gemini_agent = gemini.load_agent(runtime_settings=runtime_settings)
guardrails_agent = guardrails.load_agent(runtime_settings=runtime_settings)
function_calling_agent = function_calling.load_agent(runtime_settings=runtime_settings)
semantic_router_agent = semantic_router.load_agent(runtime_settings=runtime_settings)
task_planner_agent = task_planner.load_agent(runtime_settings=runtime_settings)
qna_agent = qna.load_agent(store=index_store, runtime_settings=runtime_settings)

# setup each agent during server startup


@contextlib.asynccontextmanager
async def lifespan(_app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
    """Setup each agent during server startup."""

    await gemini_agent.setup()
    await guardrails_agent.setup()
    await function_calling_agent.setup()
    await semantic_router_agent.setup()
    await task_planner_agent.setup()
    await qna_agent.setup()

    await store.setup_store(index_store)

    yield

    await store.cleanup_store(index_store)


app = fastapi.FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> int:
    """Root endpoint."""
    return 200


@app.get("/health")
async def health() -> int:
    """Health endpoint."""
    return 200


# register store router

app.include_router(
    router=fastapi_app.build_store_router(
        store=index_store,
        router=fastapi.APIRouter(
            prefix="/store",
            tags=["Store"],
        ),
    )
)

# register agent routers

app.include_router(
    router=fastapi_app.build_agent_router(
        agent=gemini_agent,
        router=fastapi.APIRouter(
            prefix="/gemini",
            tags=["Gemini Chat"],
        ),
    ),
)

app.include_router(
    router=fastapi_app.build_agent_router(
        agent=guardrails_agent,
        router=fastapi.APIRouter(
            prefix="/gemini-with-guardrails",
            tags=["Gemini with Guardrails"],
        ),
    ),
)

app.include_router(
    router=fastapi_app.build_agent_router(
        agent=function_calling_agent,
        router=fastapi.APIRouter(
            prefix="/function-calling",
            tags=["Function Calling"],
        ),
    ),
)

app.include_router(
    router=fastapi_app.build_agent_router(
        agent=semantic_router_agent,
        router=fastapi.APIRouter(
            prefix="/semantic-router",
            tags=["Semantic Router"],
        ),
    ),
)

app.include_router(
    router=fastapi_app.build_agent_router(
        agent=task_planner_agent,
        router=fastapi.APIRouter(
            prefix="/task-planner",
            tags=["Task Planner"],
        ),
    ),
)

app.include_router(
    router=fastapi_app.build_agent_router(
        agent=qna_agent,
        router=fastapi.APIRouter(
            prefix="/qna",
            tags=["Document Q&A"],
        ),
    ),
)
