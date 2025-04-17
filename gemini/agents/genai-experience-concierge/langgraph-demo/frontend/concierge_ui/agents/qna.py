# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""Streamlit demo page builder to avoid duplicating code."""
# disable duplicate code since chat handlers for each agent may be very similar
# pylint: disable=duplicate-code

import json
import logging
import time
from typing import Any, Generator
from concurrent.futures import ThreadPoolExecutor, Future
from threading import currentThread
import uuid


from concierge_ui import auth, remote_settings, store
from langgraph import config as lg_config
from langgraph.pregel import remote
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import (
    add_script_run_ctx,
    get_script_run_ctx,
)


logger = logging.getLogger(__name__)


def demo_builder(
    demo_id: str,
    namespace_prefix: tuple[str, ...],
    store_config: remote_settings.StoreConfig,
    left_config: remote_settings.RemoteAgentConfig,
    right_config: remote_settings.RemoteAgentConfig,
) -> None:
    session_key = f"{demo_id}-session"
    history_key = f"{demo_id}-history"
    left_thread_key = f"{session_key}-left"
    right_thread_key = f"{session_key}-right"

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
        value="""
You are a Q&A chat assistant that can answer questions about documents stored
in a vector search database. Always use your search tool provided to search
for documents before answering any query.

Not all retrieved documents may be relevant to the user query. If the content
of the documents is not enough to answer the user's question, explain that
you were unable to find enough information to provide an informed response.
""".strip(),
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

    # Initialize graphs

    left_graph = remote.RemoteGraph(
        left_config.agent_id,
        url=str(left_config.base_url),
        headers=auth.get_auth_headers(left_config),
    )

    right_graph = remote.RemoteGraph(
        right_config.agent_id,
        url=str(right_config.base_url),
        headers=auth.get_auth_headers(right_config),
    )

    left_col, right_col = st.columns(2)

    column_matrix = [
        (left_col, left_thread_key, left_graph),
        (right_col, right_thread_key, right_graph),
    ]

    # SxS chat
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = dict[str, Future]()
        for col, col_key, graph in column_matrix:
            with col:
                st.markdown(f"Thread ID: {st.session_state[col_key]}")

                # Controls
                model_name = st.selectbox(
                    "Select Model",
                    ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro-preview-03-25"],
                    key=f"{col_key}_gemini_model",
                    index=0,
                )
                temperature = st.slider("Temperature", 0.0, 2.0, 0.3, key=f"{col_key}_temperature")
                disable_tools = st.checkbox(
                    "Disable Document Search", value=False, key=f"{col_key}_disable_tools"
                )

                # Display chat messages from history on app rerun
                for message in st.session_state[history_key]:
                    with st.chat_message("user"):
                        st.write(message["user"])

                    if col_key in message:
                        with st.chat_message("assistant"):
                            st.write(message[col_key])

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
                        generator=stream_response(
                            graph=graph,
                            message=prompt,
                            config=runnable_config,
                        ),
                        col=col,
                        ctx=get_script_run_ctx(),
                    )

                    responses[col_key] = res_future

        if responses:
            # Add to chat history
            st.session_state[history_key].append(
                {
                    "user": prompt,
                    "timestamp": time.time(),
                    **{col_key: future.result() for col_key, future in responses.items()},
                }
            )


def stream_response(
    graph: remote.RemoteGraph, message: str, config: lg_config.RunnableConfig
) -> Generator[str, None, None]:
    """
    Handles chat interactions for a function calling agent by streaming responses from a remote LangGraph.

    This function takes a user message and a thread ID, and streams responses from a remote LangGraph.
    It parses the streamed chunks, which can contain text responses, function calls, function responses, or errors,
    and formats them into a human-readable text stream.

    Args:
        message (str): The user's input message.
        thread_id (str): The ID of the chat thread.

    Yields:
        str: Formatted text chunks representing text responses, function calls, function responses, or errors.
    """
    current_source = last_source = None
    for _, chunk in graph.stream(
        input={"current_turn": {"user_input": message}},
        config=config,
        stream_mode=["custom"],
    ):
        assert isinstance(chunk, dict), "Expected dictionary chunk"

        text = ""

        if "text" in chunk:
            text = chunk["text"]
            current_source = "text"

        elif "function_call" in chunk:
            function_call_dict = chunk["function_call"]

            fn_name = function_call_dict.get("name") or "unknown"
            fn_args = function_call_dict.get("args") or {}

            fn_args_string = ", ".join(f"{k}={v}" for k, v in fn_args.items())
            fn_string = f"**{fn_name}**({fn_args_string})"

            text = f"Calling function... {fn_string}"
            current_source = "function_call"

        elif "function_response" in chunk:
            function_response_dict = chunk["function_response"]

            fn_name = function_response_dict.get("name") or "unknown"

            if function_response_dict.get("response") is None:
                text = f"Received empty function response (name={fn_name})."

            elif "result" in function_response_dict.get("response"):
                fn_result = function_response_dict["response"]["result"]
                text = "\n\n".join(
                    [
                        f"Function result for **{fn_name}**...",
                        "```json",
                        json.dumps(fn_result, indent=2),
                        "```",
                    ]
                )

            elif "error" in function_response_dict.get("response"):
                fn_result = function_response_dict["response"]["error"]
                text = f"Function error (name={fn_name})... {fn_result}"

            current_source = "function_response"

        elif "error" in chunk:
            text = chunk["error"]
            current_source = "error"

        else:
            logger.warning(f"unhandled chunk case: {chunk}")

        if last_source is not None and last_source != current_source:
            text = "\n\n---\n\n" + text

        last_source = current_source

        yield text


def stream_turn_in_column(prompt, generator, col, ctx) -> str | list[Any]:
    """Execute generator inside thread context."""

    add_script_run_ctx(currentThread(), ctx)

    with col:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            res = st.write_stream(generator)

    return res
