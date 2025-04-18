# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Streamlit demo page builder to avoid duplicating code."""
# disable duplicate code since chat handlers for each agent may be very similar
# pylint: disable=duplicate-code

import logging
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor, Future
from threading import currentThread
import uuid


from concierge_ui import auth, remote_settings, store
from concierge_ui.agents import function_calling
from langgraph import config as lg_config
from langgraph.pregel import remote
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import (
    add_script_run_ctx,
    get_script_run_ctx,
)


logger = logging.getLogger(__name__)

JUDGE_MODEL_NAME = "gemini-2.0-flash-001"
QNA_MODEL_OPTIONS = [
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
]


DEFAULT_QNA_PROMPT = """
You are a Q&A chat assistant that can answer questions about documents stored
in a vector search database. Always use your search tool provided to search
for documents before answering any query.

Not all retrieved documents may be relevant to the user query. If the content
of the documents is not enough to answer the user's question, explain that
you were unable to find enough information to provide an informed response.
""".strip()

JUDGE_SYSTEM_PROMPT = """
You are an AI Bot who acts as a judge to analyze two responses to the same question. 
Please analyze both responses and provide a clear, concise judgment
indicating which response is better and why.

Your judgment should clearly state either which response is better,
followed by a brief explanation. If both responses are equal in quality, clearly indicate this.
""".strip()

JUDGE_CONTENT_TEMPLATE = """
Evaluate the following responses to the same question. Note that each response will contain
any trace of research done to get to the answer. Those research traces should be used as
the ground truth context.

QUESTION: {question}

Response A (Model on the Left):
{left_response}

Response B (Model on the Right):
{right_response}

Which one more accurately responds to the question using the source of truth?
Make sure your verdict is based on each model's strict adherence to the source of truth.
""".strip()


def demo_builder(
    demo_id: str,
    namespace_prefix: tuple[str, ...],
    store_config: remote_settings.StoreConfig,
    qna_config: remote_settings.RemoteAgentConfig,
    judge_config: remote_settings.RemoteAgentConfig,
) -> None:
    session_key = f"{demo_id}-session"
    history_key = f"{demo_id}-history"
    left_thread_key = f"{session_key}-left"
    right_thread_key = f"{session_key}-right"

    # Initialize graphs

    qna_graph = remote.RemoteGraph(
        qna_config.agent_id,
        url=str(qna_config.base_url),
        headers=auth.get_auth_headers(qna_config),
    )

    judge_graph = remote.RemoteGraph(
        judge_config.agent_id,
        url=str(judge_config.base_url),
        headers=auth.get_auth_headers(judge_config),
    )

    # Set initial IDs
    if session_key not in st.session_state:
        st.session_state[session_key] = uuid.uuid4().hex
    if left_thread_key not in st.session_state:
        st.session_state[left_thread_key] = uuid.uuid4().hex
    if right_thread_key not in st.session_state:
        st.session_state[right_thread_key] = uuid.uuid4().hex

    # Initialize chat history
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    if st.button(
        "New Session (will delete all ingested PDFs)",
        type="primary",
        icon="🔄",
        use_container_width=True,
    ):
        st.session_state[session_key] = uuid.uuid4().hex
        st.session_state[left_thread_key] = uuid.uuid4().hex
        st.session_state[right_thread_key] = uuid.uuid4().hex
        st.session_state[history_key] = []

    if st.button(
        "Clear Conversation History",
        type="secondary",
        icon="🗑️",
        use_container_width=True,
    ):
        st.session_state[left_thread_key] = uuid.uuid4().hex
        st.session_state[right_thread_key] = uuid.uuid4().hex
        st.session_state[history_key] = []

    system_prompt = st.text_area(
        "System Prompt (make sure to apply system prompt changes before submitting user input)",
        value=DEFAULT_QNA_PROMPT,
    )

    session_namespace = namespace_prefix + (st.session_state[session_key],)

    # Ingest PDFs
    with st.sidebar:
        st.header("📁 Document Setup")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Index Documents"):
            if pdf_docs:
                try:
                    with st.status("📄 Indexing documents...", expanded=True) as status:
                        for pdf in pdf_docs:
                            store.upload_pdf(
                                pdf=pdf,
                                namespace=session_namespace,
                                config=store_config,
                            )
                        status.update(label="✅ Document processing complete!", state="complete")
                except Exception as e:
                    st.error(f"Document processing failed: {str(e)}")
                else:
                    st.success("Documents ready for analysis!")
            else:
                st.warning("Please upload PDF files first!")

    # User input
    prompt = st.chat_input("Ask your question...")

    left_col, right_col = st.columns(2)

    column_matrix = [(left_col, left_thread_key), (right_col, right_thread_key)]

    # SxS chat
    judge_message = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = dict[str, Future]()
        for col, col_key in column_matrix:
            with col:
                st.markdown(f"Thread ID: {st.session_state[col_key]}")

                # Controls
                model_name = st.selectbox(
                    "Select Model",
                    QNA_MODEL_OPTIONS,
                    key=f"{col_key}_gemini_model",
                    index=0,
                )
                temperature = st.slider("Temperature", 0.0, 2.0, 0.2, key=f"{col_key}_temperature")
                disable_tools = st.checkbox(
                    "Disable Document Search", value=False, key=f"{col_key}_disable_tools"
                )

                # Display chat messages from history on app rerun
                for turn in st.session_state[history_key]:
                    with st.chat_message("user"):
                        st.write(turn["user"])

                    if col_key in turn:
                        with st.chat_message("assistant"):
                            st.write(turn[col_key])

                # Accept user input and stream responses concurrently
                if prompt:
                    runnable_config = lg_config.RunnableConfig(
                        configurable={
                            "thread_id": st.session_state[col_key],
                            "chat_config": {
                                "system_prompt": system_prompt,
                                "chat_model_name": model_name,
                                "temperature": temperature,
                                "disable_tools": disable_tools,
                            },
                            "qna_config": {
                                "namespace": list(session_namespace),
                                "text_field": store_config.retrieval_text_field,
                            },
                        }
                    )

                    res_future = executor.submit(
                        stream_turn_in_column,
                        prompt=prompt,
                        generator=function_calling.chat_handler(
                            graph=qna_graph,
                            message=prompt,
                            runnable_config=runnable_config,
                        ),
                        col=col,
                        ctx=get_script_run_ctx(),
                    )

                    responses[col_key] = res_future

        if responses:
            st.session_state[history_key].append(
                {
                    "user": prompt,
                    "timestamp": time.time(),
                    **{col_key: future.result() for col_key, future in responses.items()},
                }
            )

    if st.session_state[history_key]:
        if st.button("Compare Responses", type="primary", use_container_width=True):
            st.markdown("## LLM Judge Analysis")

            # set judge message for comparison of latest turn
            turn = st.session_state[history_key][-1]
            judge_message = JUDGE_CONTENT_TEMPLATE.format(
                question=turn["user"],
                left_response=turn[left_thread_key],
                right_response=turn[right_thread_key],
            )

            # stream judge response
            with st.chat_message("assistant"):
                st.write_stream(
                    function_calling.chat_handler(
                        graph=judge_graph,
                        message=judge_message,
                        runnable_config=lg_config.RunnableConfig(
                            configurable={
                                "thread_id": st.session_state[session_key],
                                "chat_config": {
                                    "system_prompt": JUDGE_SYSTEM_PROMPT,
                                    "chat_model_name": JUDGE_MODEL_NAME,
                                    "temperature": 0.2,
                                    "disable_tools": True,
                                },
                            }
                        ),
                    )
                )


def stream_turn_in_column(prompt, generator, col, ctx) -> str | list[Any]:
    """Execute generator inside thread context."""

    add_script_run_ctx(currentThread(), ctx)

    with col:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            res = st.write_stream(generator)

    return res
