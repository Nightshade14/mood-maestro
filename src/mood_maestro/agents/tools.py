# src/mood_maestro/agents/tools.py
import functools
import re
import json
import logging
import math
from datetime import datetime, timezone

import numpy as np
from openai import AzureOpenAI
from pymongo.database import Database
from pymongo import DESCENDING
from bson.objectid import ObjectId

# Import the shared clients and collection names from config.py
from .config import (
    ALBUMS_COLLECTION,
    ARTISTS_COLLECTION,
    GENRES_COLLECTION,
    TRACKS_COLLECTION,
    USERS_COLLECTION,
    AZURE_DEPLOYMENT,
    get_db_client,
    get_openai_client,
)

logger = logging.getLogger(__name__)


def search_track_by_name_and_artist(
    db: Database, track_name: str, artist_name: str = None
) -> dict:
    """
    Helper function to search for tracks with flexible matching.

    If the user specifies a track by a certain artist, then we find that track with both: track name and artist name.
    If the user only specifies a track name, then we find that track with just the track name. But there might be multiple tracks with the same name.
    So, we find the track with the highest popularity score. This score is present in the track document for each track. You can write a boosted search query to find the track with the highest popularity score.

    Args:
        db: MongoDB database client
        track_name: Name of the track
        artist_name: Name of the artist (optional)

    Returns:
        dict: Track document if found, None otherwise
    """
    collection = db[TRACKS_COLLECTION]

    # Build a safe, case-insensitive exact-match regex for the track name
    tn_regex = {"$regex": f"^{re.escape(track_name)}$", "$options": "i"}

    if artist_name:
        # Prefer exact artist name match (case-insensitive), then fall back to partial
        an_exact_regex = {"$regex": f"^{re.escape(artist_name)}$", "$options": "i"}
        an_partial_regex = {"$regex": re.escape(artist_name), "$options": "i"}

        # 1) Exact track + exact artist, sorted by popularity desc
        query = {"track_name": tn_regex, "artists": an_exact_regex}
        cursor = (
            collection.find(
                query, {"embedding": 1, "track_name": 1, "artists": 1, "popularity": 1}
            )
            .sort([("popularity", DESCENDING)])
            .limit(1)
        )
        result = next(cursor, None)
        if result:
            return result

        # 2) Exact track + partial artist, sorted by popularity desc
        query = {"track_name": tn_regex, "artists": an_partial_regex}
        cursor = (
            collection.find(
                query, {"embedding": 1, "track_name": 1, "artists": 1, "popularity": 1}
            )
            .sort([("popularity", DESCENDING)])
            .limit(1)
        )
        result = next(cursor, None)
        if result:
            return result

    # 3) Track-only: pick the most popular exact-name match
    query = {"track_name": tn_regex}
    cursor = (
        collection.find(
            query, {"embedding": 1, "track_name": 1, "artists": 1, "popularity": 1}
        )
        .sort([("popularity", DESCENDING)])
        .limit(1)
    )
    return next(cursor, None)


def get_entity_embedding(
    entity_name: str, entity_type: str, db: Database
) -> list[float]:
    """
    Retrieves the embedding vector for a given entity ('track', 'artist', 'genre', 'album').

    Args:
        entity_name (str): The name of the entity (e.g., "Bohemian Rhapsody", "Queen", "Thunder Imagine Dragons").
        entity_type (str): The type of the entity ('track', 'artist', 'genre', 'album').
        db (Database): The MongoDB database client.

    Returns:
        list[float]: The embedding vector for the entity. Returns None if not found.
    """
    entity_collection_map = {
        "track": TRACKS_COLLECTION,
        "song": TRACKS_COLLECTION,  # Alias for track
        "music": TRACKS_COLLECTION,  # Alias for track
        "artist": ARTISTS_COLLECTION,
        "genre": GENRES_COLLECTION,
        "album": ALBUMS_COLLECTION,
    }
    collection_name = entity_collection_map.get(entity_type.lower())
    if not collection_name:
        raise ValueError(f"Unsupported entity type: {entity_type}")

    # Handle tracks differently since they have different field structure
    if entity_type.lower() in ["track", "song", "music"]:
        logger.info(f"Searching for track: '{entity_name}'")

        # Try to parse "track_name artist_name" format
        parts = entity_name.strip().split()
        if len(parts) >= 2:
            # Try different combinations of track name and artist
            for i in range(1, len(parts)):
                track_name = " ".join(parts[:i])
                artist_name = " ".join(parts[i:])

                logger.info(f"Trying track_name='{track_name}', artist='{artist_name}'")

                result = search_track_by_name_and_artist(db, track_name, artist_name)
                if result:
                    logger.info(
                        f"Found track '{result.get('track_name')}' by {result.get('artists')} with embedding of length {len(result['embedding'])}"
                    )
                    return result["embedding"]

        # If no artist provided or not found, try just the track name
        logger.info(f"Trying track_name only: '{entity_name}'")
        result = search_track_by_name_and_artist(db, entity_name)

        if result:
            logger.info(
                f"Found track '{result.get('track_name')}' by {result.get('artists')} with embedding of length {len(result['embedding'])}"
            )
            return result["embedding"]
        else:
            logger.warning(f"No track found for '{entity_name}'")
            return None
    else:
        # For other entities (artist, genre, album), use entity_name field
        result = db[collection_name].find_one(
            {"entity_name": entity_name}, {"embedding": 1}
        )
        return result["embedding"] if result else None


def get_user_data(user_id: str, db: Database) -> dict:
    """
    Retrieves embeddings data for a specific user.

    Args:
        user_id (str): The unique identifier for the user.
        db (Database): The MongoDB database client.

    Returns:
        dict: A dictionary containing user information, such as their
              composite track embedding and listening history for score calculation.
              Example: {
                        "name": str,
                        "tracks_embedding": list[float],
                        "genres_embedding": list[float],
                        "artists_embedding": list[float],
                        "albums_embedding": list[float]
                        }
            Note: These are the exact field names used in the database.
    """
    doc = db[USERS_COLLECTION].find_one({"_id": ObjectId(user_id)}) or {}

    # Standardized field names per docstring
    name = doc.get("name") or "Anonymous User"

    # Accept legacy keys and map to new standardized keys
    tracks_embedding = doc.get("tracks_embedding") or []
    genres_embedding = doc.get("genres_embedding") or []
    artists_embedding = doc.get("artists_embedding") or []
    albums_embedding = doc.get("albums_embedding") or []

    response = {
        "name": name,
        "tracks_embedding": tracks_embedding,
        "genres_embedding": genres_embedding,
        "artists_embedding": artists_embedding,
        "albums_embedding": albums_embedding,
    }

    return response


def execute_search_pipeline(pipeline: list[dict], db: Database) -> list[dict]:
    """
        Executes a MongoDB aggregation pipeline.

        This function is the primary tool for finding candidate tracks.

        Here are some of the details needed for querying the vector index:
        - queryVector: The vector to search for.
        - path: The field to search in. = "embedding"
        - limit: The number of results to return. = 50
        - numCandidates: The number of candidates to consider. = 100
        - index: The name of the vector index to use. = "vector_index"

        Here is a sample aggregation pipeline from mongodb documentation:

        db.tracks.aggregate([
      {
        "$vectorSearch": {
          "index": "vector_index",
          "path": "embedding",
          "queryVector": [<array-of-numbers>],
          "numCandidates": <number-of-candidates>,
          "limit": <number-of-results>
        }
      }
    ])

        Args:
            pipeline (list[dict]): A valid MongoDB aggregation pipeline, typically including
                                   $match, $vectorSearch, and $project stages.
            db (Database): The MongoDB database client.

        Returns:
            list[dict]: A list of documents (tracks) matching the pipeline criteria.
                        Each document includes metadata and the vector search similarity_score.
    """
    return list(db[TRACKS_COLLECTION].aggregate(pipeline))


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
    user_vec = np.array(user_track_embedding)
    track_vecs = np.array(track_embeddings)
    user_vec_norm = user_vec / np.linalg.norm(user_vec)
    track_vecs_norm = track_vecs / np.linalg.norm(track_vecs, axis=1)[:, np.newaxis]
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
    scores, k, base_midpoint = [], 0.1, 15
    current_ts_ms = datetime.now(timezone.utc).timestamp() * 1000

    for track in tracks:
        skip_count = track.get("skip_count", 0)
        if skip_count == 0:
            scores.append(0.0)
            continue

        finish_count = track.get("finish_count", 0)
        ratio = finish_count / skip_count
        last_skip_ts_ms = track.get("last_skip_timestamp")
        sigmoid_factor = 1.0

        if last_skip_ts_ms:
            days_since_last_skip = (current_ts_ms - last_skip_ts_ms) / (
                1000 * 3600 * 24
            )
            cooldown_level = track.get("cooldown_level", 0)
            x_0 = base_midpoint * (cooldown_level + 1)
            sigmoid_factor = 1 / (1 + math.exp(-k * (days_since_last_skip - x_0)))
        score = ratio * sigmoid_factor
        if track.get("liked", False):
            score *= 1.5
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
            model=AZURE_DEPLOYMENT,
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


def submit_final_playlist(ranked_tracks: list[dict]) -> list[dict]:
    """
    Call this function as the absolute final step to submit the ranked playlist.
    The input should be the final, sorted list of track dictionaries.
    """
    return ranked_tracks


# --- Create Agent-Ready Tools using functools.partial ---
# This "bakes in" the shared clients, so the agent doesn't need to manage them.
shared_db_client = get_db_client()

get_entity_embedding_tool = functools.partial(get_entity_embedding, db=shared_db_client)
get_user_data_tool = functools.partial(get_user_data, db=shared_db_client)
execute_search_pipeline_tool = functools.partial(
    execute_search_pipeline, db=shared_db_client
)
search_track_by_name_and_artist_tool = functools.partial(
    search_track_by_name_and_artist, db=shared_db_client
)

shared_azure_client = get_openai_client()

determine_ranking_weights_tool = functools.partial(
    determine_ranking_weights, client=shared_azure_client
)

agent_tool_list = [
    get_entity_embedding_tool,
    get_user_data_tool,
    execute_search_pipeline_tool,
    calculate_personalization_scores,
    calculate_reengagement_scores,
    determine_ranking_weights_tool,
    rank_tracks,
    submit_final_playlist,
    search_track_by_name_and_artist_tool,
]
