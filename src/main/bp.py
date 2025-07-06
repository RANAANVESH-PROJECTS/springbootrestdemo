import os
import asyncio
import re # Import regex module
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from browser_use import Agent, BrowserSession
from browser_use.llm import ChatOpenAI as BrowserLLM
from langchain_openai import ChatOpenAI as LCChatOpenAI
from langchain_core.runnables.base import coerce_to_runnable
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Load credentials from .env
load_dotenv()

print("Environment variables loaded.")
# --- TEMPORARY DEBUGGING PRINT ---
debug_amazon_password = os.getenv("AMAZON_PASSWORD")
print(f"DEBUG: Value of AMAZON_PASSWORD seen by script: {'*' * len(debug_amazon_password) if debug_amazon_password else 'None/Empty'}")
# --- END TEMPORARY DEBUGGING PRINT ---


# Flask app
app = Flask(__name__)

# LLMs
browser_llm = BrowserLLM(model="gpt-4o")
coord_llm = coerce_to_runnable(LCChatOpenAI(model="gpt-4o"))

# --- Tool for Login and Browsing (requires credentials) ---
def login_and_browse_tool_fn(query: str) -> str:
    print("[LOGIN_BROWSE_TOOL] Called with query:", query)

    # Regex to find "username key <KEY_NAME>" and "password key <KEY_NAME>"
    username_key_match = re.search(r"username key (\w+)", query, re.IGNORECASE)
    password_key_match = re.search(r"password key (\w+)", query, re.IGNORECASE)

    username_env_var_name = username_key_match.group(1) if username_key_match else None
    password_env_var_name = password_key_match.group(1) if password_key_match else None

    if not username_env_var_name or not password_env_var_name:
        return "Error: This tool requires both username key and password key in the query (e.g., 'username key GMAIL_EMAIL and password key GMAIL_PASSWORD')."

    username_value = os.getenv(username_env_var_name)
    password_value = os.getenv(password_env_var_name)

    if not username_value:
        return f"Error: Environment variable '{username_env_var_name}' not found or empty in .env file."
    if not password_value:
        return f"Error: Environment variable '{password_env_var_name}' not found or empty in .env file."

    print(f"Fetched username value for '{username_env_var_name}': {username_value}")
    print(f"Fetched password value for '{password_env_var_name}': {'*' * len(password_value)}") # Mask password for logging

    # Prepare the base task by removing the key directives from the original query
    base_task = re.sub(r"username key \w+", "", query, flags=re.IGNORECASE).strip()
    base_task = re.sub(r"password key \w+", "", base_task, flags=re.IGNORECASE).strip()

    # Inject the actual username and password values directly into the agent_task.
    # This forces the browser_llm to use the real credentials.
    agent_task = (
        f"For the username/email input field, you MUST use '{username_value}' as the value. "
        f"For the password input field, you MUST use '{password_value}' as the value. "
        "Use these exact strings in the 'value' field of your browser steps. "
        "Respond ONLY with a JSON array of browser steps. No explanation.\n\n"
        f"{base_task}"
    )
    print(f"[LOGIN_BROWSE_TOOL] Agent task sent to browser_llm:\n{agent_task}")

    # Initialize BrowserSession without sensitive_data, as values are now directly in agent_task.
    session = BrowserSession(headless=False, sensitive_data={}) # Clear sensitive_data
    print("[LOGIN_BROWSE_TOOL] Initialized BrowserSession without sensitive data (values are in task).")

    # Pass the explicitly instructed task (with real values) to the Agent
    agent = Agent(task=agent_task, llm=browser_llm, browser_session=session)

    print("[LOGIN_BROWSE_TOOL] Planning browser steps...")
    # Attempt to get a planned list of steps.
    steps = asyncio.run(agent._run_planner())

    if not steps:
        print("⚠️ Planner returned None. Falling back to direct run...")
        # If planner fails, fall back to agent.run().
        # In this case, the LLM was already prompted with the real values.
        return asyncio.run(agent.run())

    print("[LOGIN_BROWSE_TOOL] Raw steps from planner (should contain actual username/password):\n", steps)

    # No manual substitution loop needed here, as the LLM was prompted with actual values.
    return asyncio.run(agent.execute(steps))

# Register the login and browse tool
login_and_browse_tool = Tool(
    name="login_and_browse_tool",
    func=login_and_browse_tool_fn,
    description="Automate browser tasks that require logging into a website. "
                "Always specify credentials by including 'username key ENV_VAR_NAME and password key ENV_VAR_NAME' "
                "in the query, where ENV_VAR_NAME is the name of the environment variable holding the credential. "
                "Example: 'Sign in to Gmail using username key GMAIL_EMAIL and password key GMAIL_PASSWORD'."
)

# --- New Tool for General Browsing (no credentials required) ---
def general_browse_tool_fn(query: str) -> str:
    print("[GENERAL_BROWSE_TOOL] Called with query:", query)

    # For general browsing, no sensitive data is needed in the session.
    session = BrowserSession(headless=False, sensitive_data={})
    print("[GENERAL_BROWSE_TOOL] Initialized BrowserSession for general browsing.")

    # The agent's task is simply the user's query.
    agent = Agent(task=query, llm=browser_llm, browser_session=session)

    print("[GENERAL_BROWSE_TOOL] Starting general browsing task...")
    # For general browsing, we can directly run the agent with the query.
    return asyncio.run(agent.run())

# Register the general browsing tool
general_browse_tool = Tool(
    name="general_browse_tool",
    func=general_browse_tool_fn,
    description="Automate general browser tasks that do NOT require logging into a website. "
                "Use this tool for navigating to URLs, searching for information, or reading content on public websites. "
                "Example: 'Go to example.com and find the contact information'."
)


# Coordinator agent
# Custom prefix to guide the agent's thought process
agent_prefix = """You are an AI assistant capable of interacting with web browsers.
You have access to two tools: 'login_and_browse_tool' and 'general_browse_tool'.
- Use 'login_and_browse_tool' ONLY when the user explicitly asks to log in to a website and provides 'username key' and 'password key' in the query.
- Use 'general_browse_tool' for all other browsing tasks that do not involve logging in or require credentials.
Carefully analyze the user's request to determine which tool is appropriate.
"""

coordinator_agent = initialize_agent(
    tools=[login_and_browse_tool, general_browse_tool],
    llm=coord_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": agent_prefix
    }
)

# API endpoint
@app.route("/browser-agent", methods=["POST"])
def browser_agent_endpoint():
    query = request.get_json().get("query", "")
    # The coordinator agent will now decide which tool to use.
    result = coordinator_agent.invoke(query)
    return jsonify({"result": result["output"]})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
