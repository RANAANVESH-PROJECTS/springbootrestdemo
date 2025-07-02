from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from custom_llm_wrapper import get_llm_client_and_headers, call_internal_llm

# Get internal Gemini-compatible client and headers
client, headers = get_llm_client_and_headers()

# Define a wrapper function to be used as the LLM interface
def internal_llm_chat_fn(agent, messages, sender, config):
    query_text = messages[-1]["content"]
    response = call_internal_llm(client, headers, query_text)
    return True, response

# User agent (no human input, uses API prompt)
user = UserProxyAgent(name="User", human_input_mode="NEVER")

# Assistant agent powered by internal Gemini API
assistant = AssistantAgent(
    name="BrowserAgent",
    llm_config={
        "functions": [],
        "chat": internal_llm_chat_fn,
    }
)

# Create group chat with manager
group_chat = GroupChat(
    agents=[user, assistant],
    messages=[],
    max_round=5
)

manager = GroupChatManager(groupchat=group_chat)

# Kick off the conversation (you can update the message below)
user.initiate_chat(
    manager,
    message="Go to https://amazon.in and search for Canon camera"
)
