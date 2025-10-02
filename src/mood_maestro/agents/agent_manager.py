# src/mood_maestro/agents/agent_manager.py
import functools
import inspect
import json
import autogen
from .config import LLM_CONFIG
from .tools import agent_tool_list


def run_recommendation_flow(query: str, user_id: str) -> list[dict]:
    """
    Sets up and runs the AutoGen multi-agent workflow for music recommendation.

    This function orchestrates a conversation between a Planner, a MusicQueryAgent (worker),
    and a CodeExecutorAgent to translate a natural language query into a ranked playlist.
    """

    # Define the Agents with specific roles (System Prompts)
    planner = autogen.AssistantAgent(
        name="PlannerAgent",
        system_message=f"""You are a master planner for a music recommendation system.
        Your job is to take a user's request and create a clear, step-by-step plan.
        Do NOT implement the steps yourself - delegate to the MusicQueryAgent.
        
        The user you are serving has the ID: {user_id}
        
        Available tools for the MusicQueryAgent to use:
        - get_user_data(user_id): Get user listening history and preferences
        - get_entity_embedding(entity_name, entity_type): Get embedding for a track/artist/genre/album
        - execute_search_pipeline(pipeline): Run MongoDB aggregation for finding tracks
        - calculate_personalization_scores(track_embeddings, user_track_embedding): Calculate user-track similarity
        - calculate_reengagement_scores(tracks): Calculate re-engagement scores
        - determine_ranking_weights(query): Use LLM to determine ranking weights
        - rank_tracks(tracks_with_scores, weights): Rank tracks by weighted scores
        - submit_final_playlist(ranked_tracks): Submit the final playlist (MUST be called last)
        
        Create a plan that uses these tools appropriately. The final step MUST call submit_final_playlist.
        """,
        llm_config=LLM_CONFIG,
    )

    music_query_agent = autogen.AssistantAgent(
        name="MusicQueryAgent",
        system_message="""You are a music recommendation specialist.
        You will be given a plan. For each step, call the appropriate registered function tools.
        Use the function calling interface, NOT Python code blocks.
        Available functions: get_entity_embedding, get_user_data, execute_search_pipeline, 
        calculate_personalization_scores, calculate_reengagement_scores, determine_ranking_weights,
        rank_tracks, submit_final_playlist.
        Always wait for the result of each function call before proceeding to the next step.""",
        llm_config=LLM_CONFIG,
    )

    code_executor_agent = autogen.UserProxyAgent(
        name="CodeExecutorAgent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,  # Disable code execution, use function calling instead
        system_message="""You are the function executor. Execute the function calls requested by the
        MusicQueryAgent and report back the results. The final ranked list of tracks
        is the ultimate result of the entire process.""",
    )

    # Set up the Group Chat
    groupchat = autogen.GroupChat(
        agents=[code_executor_agent, planner, music_query_agent],
        messages=[],
        max_round=20,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG)

    # Register all the prepared tools for the agents to use
    # Build function map with proper names and wrap partials
    function_map = {}
    for tool in agent_tool_list:
        # Get the original function name for registration
        if isinstance(tool, functools.partial):
            tool_name = tool.func.__name__

            # Create a wrapper function that calls the partial
            # This is needed because register_for_llm doesn't accept partials
            # We need to preserve the signature of the remaining parameters
            def create_wrapper(partial_func):
                # Get the original function's signature
                orig_func = partial_func.func
                orig_sig = inspect.signature(orig_func)

                # Get the parameters that are NOT already bound in the partial
                bound_params = set(partial_func.keywords.keys())
                remaining_params = [
                    param
                    for name, param in orig_sig.parameters.items()
                    if name not in bound_params
                ]

                # Create a new signature with only the remaining parameters
                new_sig = orig_sig.replace(parameters=remaining_params)

                # Create the wrapper function
                def wrapper(*args, **kwargs):
                    return partial_func(*args, **kwargs)

                # Set the proper attributes
                wrapper.__name__ = orig_func.__name__
                wrapper.__doc__ = orig_func.__doc__ or ""
                wrapper.__signature__ = new_sig
                wrapper.__annotations__ = {
                    name: param.annotation for name, param in new_sig.parameters.items()
                }
                if new_sig.return_annotation != inspect.Signature.empty:
                    wrapper.__annotations__["return"] = new_sig.return_annotation

                return wrapper

            func = create_wrapper(tool)
        else:
            func = tool

        tool_name = func.__name__
        function_map[tool_name] = func

    # Register functions for both agents
    for func_name, func in function_map.items():
        music_query_agent.register_for_llm(
            name=func_name, description=func.__doc__ or ""
        )(func)
        code_executor_agent.register_for_execution(name=func_name)(func)

    # Initiate the chat and kick off the workflow
    code_executor_agent.initiate_chat(
        manager,
        message=f"Generate a playlist based on the following request: '{query}'",
    )

    # Parse the final result from the chat history
    # Look for the result of submit_final_playlist function call
    final_ranked_list = []
    for msg in reversed(groupchat.messages):
        # Check if this is a function call result from submit_final_playlist
        if msg.get("name") == "CodeExecutorAgent":
            content = msg.get("content", "")
            # Try to parse the function result
            try:
                # Look for function call results in the content
                if "submit_final_playlist" in content:
                    # Extract the result - it might be in various formats
                    if "[{" in content:
                        start_index = content.find("[{")
                        end_index = content.rfind("}]") + 2
                        if start_index != -1 and end_index > start_index:
                            list_str = content[start_index:end_index]
                            final_ranked_list = json.loads(list_str.replace("'", '"'))
                            break
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue

        # Also check for tool_responses in message
        if "tool_responses" in msg:
            for response in msg["tool_responses"]:
                if response.get("tool_call_id") and isinstance(
                    response.get("content"), list
                ):
                    final_ranked_list = response["content"]
                    break
            if final_ranked_list:
                break

    return final_ranked_list
