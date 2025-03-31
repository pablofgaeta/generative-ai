# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.

import streamlit as st

st.set_page_config(
    page_title="GenAI Experience Concierge",
    page_icon="👋",
)

st.write("# Welcome to the GenAI Experience Concierge Demo! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
The GenAI Experience Concierge demo consists of a few chat interfaces for some langgraph agents deployed as FastAPI servers.

Each agent server is compatible with the [LangGraph Cloud API](langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html). Not all functionality/endpoints are supported (especially the multi-assistant endpoints). A minimal subset of endpoints required by the [langgraph_sdk.RemoteGraph](https://langchain-ai.github.io/langgraph/reference/remote_graph/) interface. The `RemoteGraph` interface is an implementation of the `PregelProtocol`, the same protocol used by `CompiledGraph` (the local LangGraph class), making usage between locally built graphs and remote server graphs seamless.

Source code can be found at: [https://gitlab.com/google-pso/ais/genai/genai-experience-concierge](https://gitlab.com/google-pso/ais/genai/genai-experience-concierge)

**👈 Select a demo from the sidebar** to see some examples of what the GenAI Experience Concierge can do!
"""
)
