# src/mood_maestro/agents/config.py
import os

from openai import AzureOpenAI
from pymongo import MongoClient

# --- Azure OpenAI Client ---
shared_azure_client = AzureOpenAI(api_version="2024-02-01")

# --- MongoDB Client ---
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
if not mongo_uri or not db_name:
    raise ValueError("MONGO_URI or DB_NAME not found in environment variables.")

print("Initializing shared MongoDB client...")
mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
# Ping the server to confirm a successful connection
mongo_client.admin.command("ping")
shared_db_client = mongo_client[db_name]

# --- Environment Variable Names ---
# Store collection names for easy access
TRACKS_COLLECTION = os.getenv("MONGO_TRACKS_COLLECTION")
ARTISTS_COLLECTION = os.getenv("MONGO_ARTISTS_COLLECTION")
GENRES_COLLECTION = os.getenv("MONGO_GENRES_COLLECTION")
ALBUMS_COLLECTION = os.getenv("MONGO_ALBUMS_COLLECTION")
USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION")
