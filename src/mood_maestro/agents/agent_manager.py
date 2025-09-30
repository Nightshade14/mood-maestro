# src/mood_maestro/agents/agent_manager.py
import os
import json
import autogen
from ..agents.tools import agent_tool_list

def run_recommendation_flow(query: str, user_id: str) -> list[dict]:
    """
    Sets up and runs the AutoGen multi-agent workflow for music recommendation.
    
    This function orchestrates a conversation between a Planner, a MusicQueryAgent (worker),
    and a CodeExecutorAgent to translate a natural language query into a ranked playlist.
    """
    # 1. Configure the LLM for the agents
    llm_config = {
        "config_list": [{
            "model": os.getenv("AZURE_DEPLOYMENT_NAME"),
            "api_key": os.getenv("AZURE_OPENAI_KEY"),
            "api_type": "azure",
            "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": "2024-02-01",
        }],
        "temperature": 0.0,
    }

    # 2. Define the Agents with specific roles (System Prompts)
    planner = autogen.AssistantAgent(
        name="PlannerAgent",
        system_message=f"""You are a master planner for a music recommendation system.
        Your job is to take a user's request and create a clear, step-by-step Python-based plan.
        Do NOT write the code for the steps yourself.
        The user you are serving has the ID: '{user_id}'.
        For any step requiring user data, you must use the get_user_data tool with this ID.
        Delegate the code implementation for each step to the MusicQueryAgent.
        Your final step should ALWAYS be to call the rank_tracks function and print the final ranked list.
        """,
        llm_config=llm_config,
    )

    music_query_agent = autogen.AssistantAgent(
        name="MusicQueryAgent",
        system_message="""You are a music recommendation specialist and Python programmer.
        You will be given a plan. For each step, generate the necessary Python code by calling the
        available functions. Await the result of each step before proceeding to the next.
        Do not explain the code. Only output the code block.""",
        llm_config=llm_config,
    )

    code_executor_agent = autogen.UserProxyAgent(
        name="CodeExecutorAgent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding"},
        system_message="""You are the code executor. Run the Python code provided by the
        MusicQueryAgent and report back the results. The final ranked list of tracks
        is the ultimate result of the entire process.""",
    )

    # 3. Set up the Group Chat
    groupchat = autogen.GroupChat(
        agents=[code_executor_agent, planner, music_query_agent],
        messages=[],
        max_round=20
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # 4. Register all the prepared tools for the agents to use
    for tool in agent_tool_list:
        # Get the original function name for registration
        tool_name = tool.func.__name__ if isinstance(tool, functools.partial) else tool.__name__
        autogen.agentchat.register_function(
            tool,
            caller=music_query_agent,
            executor=code_executor_agent,
            name=tool_name,
            description=tool.__doc__,
        )

    # 5. Initiate the chat and kick off the workflow
    code_executor_agent.initiate_chat(
        manager,
        message=f"Generate a playlist based on the following request: '{query}'"
    )
    
    # 6. Parse the final result from the chat history
    # The final ranked list is typically the output of the last message from the executor
    final_ranked_list = []
    for msg in reversed(groupchat.messages):
        if msg["name"] == "CodeExecutorAgent" and "execution succeeded" in msg.get("content", ""):
            # The result is often in a string literal, so we parse it carefully
            content = msg["content"]
            try:
                # Find the start of the list '[' and end ']'
                start_index = content.find("[{")
                end_index = content.rfind("}]") + 2
                if start_index != -1 and end_index != -1:
                    list_str = content[start_index:end_index]
                    final_ranked_list = json.loads(list_str.replace("'", "\"")) # Basic handling for dicts
                    break
            except (json.JSONDecodeError, SyntaxError):
                continue # Keep searching if parsing fails

    return final_ranked_list