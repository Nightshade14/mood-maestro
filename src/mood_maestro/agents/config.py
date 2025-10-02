# src/mood_maestro/agents/config.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

load_dotenv(".env")

LLM_CONFIG = {
    "config_list": [{
        "model": os.getenv("AZURE_DEPLOYMENT"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_type": "azure",
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_API_VERSION"),
    }],
    "temperature": 0.0,
}


_db_client = None

def get_db_client():
    """
    Returns a shared instance of the MongoDB database client.
    Initializes the connection on the first call.
    """
    global _db_client
    if _db_client is None:
        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("DB_NAME")
        if not mongo_uri or not db_name:
            raise ValueError("MONGO_URI or DB_NAME not found in environment variables.")
        
        print("Initializing shared MongoDB client for the first time...")
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # client.admin.command("ping") # Verify connection
            _db_client = client[db_name]
            print("MongoDB connection successful.")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Could not connect to MongoDB: {e}")
            
    return _db_client

def get_openai_client():
    """Returns an AzureOpenAI client instance based on config."""
    config = LLM_CONFIG["config_list"][0]
    return AzureOpenAI(
        azure_endpoint=config["base_url"],
        api_key=config["api_key"],
        api_version=config["api_version"]
    )


# --- Collection Names & Other Constants ---
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") 
TRACKS_COLLECTION = os.getenv("MONGO_TRACKS_COLLECTION")
ALBUMS_COLLECTION = os.getenv("MONGO_ALBUMS_COLLECTION")
ARTISTS_COLLECTION = os.getenv("MONGO_ARTISTS_COLLECTION")
GENRES_COLLECTION = os.getenv("MONGO_GENRES_COLLECTION")
USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION")