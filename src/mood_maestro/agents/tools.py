import json
import logging
import os

import numpy as np
import pymongo
from openai import AzureOpenAI
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_to_mongo(
    collection_env_name: str,
) -> tuple[MongoClient, pymongo.collection.Collection]:
    """
    Connects to MongoDB using environment variables and returns the client and specified collection.
    Args:
        collection_env_name (str): The environment variable name for the collection to connect to.
                                    E.g., "MONGO_TRACKS_COLLECTION", "MONGO_ARTISTS_COLLECTION".
    Returns:
        Tuple[MongoClient, pymongo.collection.Collection]: The MongoDB client and the specified collection.
    """

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv(collection_env_name)

    if not mongo_uri:
        raise ValueError("MONGO_URI not found in environment variables.")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[db_name]
    collection = db[collection_name]
    logger.info("Connected to MongoDB")
    return client, collection


def get_entity_embedding(entity_name: str, entity_type: str) -> list[float]:
    """
    Retrieves the embedding vector for a given entity (track, artist, genre, etc.).

    Args:
        entity_name (str): The name of the entity (e.g., "Bohemian Rhapsody", "Queen").
        entity_type (str): The type of the entity ('track', 'artist', 'genre', 'album').

    Returns:
        list[float]: The embedding vector for the entity. Returns None if not found.
    """
    # Implementation would involve a direct lookup in the corresponding MongoDB collection.
    # e.g., db.artists.find_one({"name": entity_name}, {"embedding": 1})

    entity_collection_env_map = {
        "song": "MONGO_TRACKS_COLLECTION",  # general fallback
        "music": "MONGO_TRACKS_COLLECTION",  # general fallback
        "track": "MONGO_TRACKS_COLLECTION",
        "artist": "MONGO_ARTISTS_COLLECTION",
        "genre": "MONGO_GENRES_COLLECTION",
        "album": "MONGO_ALBUMS_COLLECTION",
    }

    collection_env_name = entity_collection_env_map.get(entity_type.lower())
    if not collection_env_name:
        raise ValueError(f"Unsupported entity type: {entity_type}")

    client = None
    try:
        client, collection = connect_to_mongo(collection_env_name)
        result = collection.find_one({"entity_name": entity_name}, {"embedding": 1})
        return result["embedding"] if result else None

    except Exception as e:
        logger.error(f"Error retrieving embedding for {entity_name}: {e}")
        return None

    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


def get_user_data(user_id: str) -> dict:
    """
    Retrieves key data for a specific user.

    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        dict: A dictionary containing user information, such as their
              composite track embedding and listening history for score calculation.
              Example: {
                        "user_track_embedding": [...],
                       "user_genre_embedding": [...],
                        "user_artist_embedding": [...],
                        "user_album_embedding": [...]
                        }
    """
    # Implementation would query the 'Users' collection.
    pass
    collection_name = os.getenv("MONGO_USERS_COLLECTION")
    client = None
    try:
        client, collection = connect_to_mongo(collection_name)
        result = collection.find_one({"_id": user_id})
        return result if result else {}

    except Exception as e:
        logger.error(f"Error retrieving user data for {user_id}: {e}")
        return {}
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


def execute_search_pipeline(pipeline: list[dict]) -> list[dict]:
    """
    Executes a complex MongoDB aggregation pipeline and returns the results.
    This function is the primary tool for finding candidate tracks.

    Args:
        pipeline (list[dict]): A valid MongoDB aggregation pipeline, typically including
                               $match, $vectorSearch, and $project stages.

    Returns:
        list[dict]: A list of documents (tracks) matching the pipeline criteria.
                    Each document includes metadata and the vector search similarity_score.
    """
    # Implementation connects to MongoDB and runs the aggregation.

    with connect_to_mongo("MONGO_TRACKS_COLLECTION") as (client, collection):
        try:
            results = list(collection.aggregate(pipeline))
            return results
        except Exception as e:
            logger.error(f"Error executing search pipeline: {e}")
            return []
        finally:
            if client:
                client.close()
                logger.info("MongoDB connection closed")


import math
from datetime import datetime, timezone


def calculate_personalization_scores(
    track_embeddings: list[list[float]], user_track_embedding: list[float]
) -> list[float]:
    """
    Calculates the cosine similarity between a user's embedding and a list of track embeddings.

    Args:
        track_embeddings (list[list[float]]): A list of track embedding vectors.
        user_embedding (list[float]): The user's composite embedding vector.

    Returns:
        list[float]: A list of personalization scores, one for each track.
    """
    user_vector = np.array(user_track_embedding)
    track_vectors = np.array(track_embeddings)
    # Normalize vectors to unit length
    user_vec_norm = user_vector / np.linalg.norm(user_vector)
    track_vecs_norm = (
        track_vectors / np.linalg.norm(track_vectors, axis=1)[:, np.newaxis]
    )
    # Compute cosine similarity (dot product of normalized vectors)
    cosine_similarities = np.dot(track_vecs_norm, user_vec_norm)
    return cosine_similarities.tolist()


def calculate_reengagement_scores(tracks: list[dict]) -> list[float]:
    """
    Calculates the re-engagement score for a list of tracks.

    This score is designed to intelligently resurface tracks that a user may have
    skipped previously but might enjoy now. It balances the skip/finish ratio
    with a time-decay component based on the last skip date.

    Args:
        tracks (list[dict]): The list of candidate track documents, each containing
                             fields like 'skip_count', 'finish_count', etc.

    Returns:
        list[float]: A list of re-engagement scores, one for each track.
    """
    scores = []
    k = 0.1  # Controls the steepness of the curve
    base_midpoint = 15  # The base number of days for the sigmoid midpoint

    current_ts_ms = datetime.now(timezone.utc).timestamp() * 1000

    for track in tracks:
        skip_count = track.get("skip_count", 0)
        finish_count = track.get("finish_count", 0)

        # If a track has never been skipped, its re-engagement score is zero.
        if skip_count == 0:
            scores.append(0.0)
            continue

        ratio = finish_count / skip_count

        last_skip_ts_ms = track.get("last_skip_timestamp")

        # If there's no skip timestamp, the time penalty doesn't apply.
        sigmoid_factor = 1.0

        if last_skip_ts_ms is not None:
            ms_since_last_skip = current_ts_ms - last_skip_ts_ms
            days_since_last_skip = ms_since_last_skip / (1000 * 3600 * 24)

            cooldown_level = track.get("cooldown_level", 0)
            # The midpoint shifts based on how often a user skips this track
            x_0 = base_midpoint * (cooldown_level + 1)

            # The sigmoid function ensures the score gently increases as more
            # time passes since the last skip.
            sigmoid_factor = 1 / (1 + math.exp(-k * (days_since_last_skip - x_0)))

        score = ratio * sigmoid_factor
        if track.get("liked", False):
            like_boost_multiplier = 1.5  # This value should be tuned
            score *= like_boost_multiplier

        scores.append(score)

    return scores


def determine_ranking_weights(query: str, client: AzureOpenAI) -> dict[str, float]:
    """
    Uses an Azure OpenAI LLM to determine ranking weights.
    The client object must be provided as an argument.
    """
    system_prompt = """
    You are an expert system for a music recommendation engine. Your task is to analyze a user's query
    and return a JSON object with weights for four scoring components: 'popularity', 'similarity',
    'personalization', and 'reengagement'.

    The weights must be floats and must sum to 1.0.

    - 'popularity': How popular/mainstream the song is. Good for discovering new hits.
    - 'similarity': How similar the song is to a specific track/artist mentioned by the user.
    - 'personalization': How well the song matches the user's overall listening habits.
    - 'reengagement': A score for bringing back songs the user skipped a while ago.

    Analyze the user's intent. For example:
    - If the query is "find songs like 'Song X'", 'similarity' should be high.
    - If the query is "show me something new and popular", 'popularity' should be high.
    - If the query is "play my usual vibe", 'personalization' should be high.
    - If the query is "surprise me with something I might have missed", 'reengagement' could be non-zero.
    
    Only return the JSON object, nothing else.
    """

    try:
        response = client.chat.completions.create(
            model="YOUR_AZURE_DEPLOYMENT_NAME",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the user's query: '{query}'"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in determine_ranking_weights: {e}")
        # Return a safe default
        return {
            "popularity": 0.25,
            "similarity": 0.25,
            "personalization": 0.5,
            "reengagement": 0.0,
        }


def rank_tracks(
    tracks_with_scores: list[dict], weights: dict[str, float]
) -> list[dict]:
    """
    Calculates the final weighted score for each track and ranks them.

    Args:
        tracks_with_scores (list[dict]): List of tracks, each with all component scores.
        weights (dict[str, float]): The weights determined by the LLM.

    Returns:
        list[dict]: The final list of tracks, sorted by the 'ranking_score' in descending order.
    """
    for track in tracks_with_scores:
        ranking_score = (
            track.get("popularity_score", 0) * weights.get("popularity", 0)
            + track.get("similarity_score", 0) * weights.get("similarity", 0)
            + track.get("personalization_score", 0) * weights.get("personalization", 0)
            + track.get("reengagement_score", 0) * weights.get("reengagement", 0)
        )
        track["ranking_score"] = ranking_score

    return sorted(
        tracks_with_scores, key=lambda x: x.get("ranking_score", 0), reverse=True
    )
