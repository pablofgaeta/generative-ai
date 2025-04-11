from google.adk.agents import LlmAgent

support_agent = LlmAgent(
    name="CustomerSupportAssistant",
    description="Answer customer service questions about the Cymbal retail company. Cymbal offers both online retail and physical stores. Feel free to make up information about this fictional company, this is just for the purposes of a demo.",
)
search_agent = LlmAgent(
    name="RetailSearchAssistant",
    description="Have a conversation with the user and answer questions about the Cymbal retail company. Cymbal offers both online retail and physical stores. Feel free to make up information about this fictional company, this is just for the purposes of a demo.",
)

root_agent = LlmAgent(
    name="CymbalRetailAssistant",
    model="gemini-2.0-flash-001",
    instruction=f"""
Route user requests:
- {search_agent.name}: Any pleasantries/general conversation or discussion of Cymbal retail products/stores/inventory, including live data.
- {support_agent.name}: Queries related to customer service such as item returns, policies, complaints, FAQs, escalations, etc.

If the request is not related to either, kindly guide the user to an appropriate topic.
""".strip(),
    description="Main help desk router.",
    sub_agents=[support_agent, search_agent],
)
