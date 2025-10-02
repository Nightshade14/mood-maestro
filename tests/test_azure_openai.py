# tests/test_azure_openai.py
from src.mood_maestro.agents.config import get_openai_client, LLM_CONFIG

# 1. Get the client using the new function
client = get_openai_client()

# 2. Get deployment name from the config
# Note: I used AZURE_DEPLOYMENT from config.py, but you can also get it from LLM_CONFIG
deployment_name = LLM_CONFIG["config_list"][0]["model"]

# Send a test prompt
def test_openai_connection():
    """
    This is now a proper test function that pytest can discover.
    """
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the primary difference between Azure OpenAI and OpenAI?"}
            ]
        )
        
        print("\nResponse from Azure OpenAI:")
        print(response.choices[0].message.content)
        
        # Add an assertion for a real test
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    except Exception as e:
        # Fail the test if an exception occurs
        assert False, f"An error occurred during API call: {e}"