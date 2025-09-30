import logging
import os
import time

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

from mood_maestro.core.models import TrackAudioFeatures, TrackDocument

# Module logger
logger = logging.getLogger("mood_maestro.set_tracks_collection")


def configure_logging_from_env() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_dataset(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {file_path}")
    return pd.read_csv(file_path)


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=["track_id", "track_name"], inplace=True)
    df.drop_duplicates(subset=["track_id"], inplace=True)
    df = df.reset_index(drop=True)
    return df


def load_env_variables() -> None:
    load_dotenv(".env")
    configure_logging_from_env()
    logger.info("Environment loaded")


def connect_to_mongo(
    db_name: str, collection_name: str
) -> tuple[MongoClient, pymongo.collection.Collection]:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not found in environment variables.")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[db_name]
    collection = db[collection_name]
    logger.info("Connected to MongoDB")
    return client, collection


def prepare_documents(df: pd.DataFrame, embeddings: np.ndarray) -> list[dict]:
    documents = []
    for index in tqdm(range(len(df)), total=len(df), desc="Preparing Documents"):
        row = df.iloc[index]
        try:
            audio = TrackAudioFeatures(
                danceability=row["danceability"],
                energy=row["energy"],
                key=int(row["key"]),
                loudness=float(row["loudness"]),
                mode=int(row["mode"]),
                speechiness=float(row["speechiness"]),
                acousticness=float(row["acousticness"]),
                instrumentalness=float(row["instrumentalness"]),
                liveness=float(row["liveness"]),
                valence=float(row["valence"]),
                tempo=float(row["tempo"]),
            )

            doc_model = TrackDocument(
                _id=str(row["track_id"]),
                track_name=row["track_name"],
                artists=[
                    a.strip() for a in str(row["artists"]).split(";") if a.strip()
                ],
                album_name=row.get("album_name"),
                track_genre=row.get("track_genre"),
                popularity=float(row["popularity"]),
                duration_ms=int(row["duration_ms"]),
                explicit=bool(row["explicit"]),
                audio_features=audio,
                embedding=embeddings[index].tolist(),
                skip_count=0,
                finish_count=0,
                liked=False,
                cooldown_level=0,
                last_skip_timestamp=None,
            )
            documents.append(doc_model.model_dump(by_alias=True))
        except Exception as e:
            logger.warning(f"Skipping row {index}: {e}", exc_info=True)
            continue
    return documents


def empty_and_populate_collection(
    collection: pymongo.collection.Collection, documents: list
) -> None:
    """Remove existing documents and upload the provided list in batches.

    Batches are of size 10,000. Each batch uses insert_many with ordered=False
    for performance and resilience. Retries a few times on transient errors.
    """
    batch_size = 50000
    max_retries = 3
    logger.info(f"Uploading {len(documents)} documents in batches of {batch_size}...")
    # Clear collection first
    collection.delete_many({})

    if not documents:
        logger.info("No documents to upload.")
        return

    total = len(documents)
    for start in tqdm(range(0, total, batch_size), desc="Uploading Batches"):
        end = min(start + batch_size, total)
        batch = documents[start:end]
        attempt = 0
        while attempt < max_retries:
            try:
                collection.insert_many(batch, ordered=True)
                logger.info(f"Uploaded batch {start + 1}-{end} ({len(batch)} docs)")
                break
            except pymongo.errors.BulkWriteError as bwe:
                # BulkWriteError can occur for duplicate _id or other write errors.
                # Log and continue if some writes succeeded.
                logger.warning(
                    f"BulkWriteError on batch {start + 1}-{end}: {bwe.details}"
                )
                # If ordered=False some docs may have been written; stop retrying this batch.
                break
            except (pymongo.errors.AutoReconnect, pymongo.errors.NetworkTimeout) as e:
                attempt += 1
                wait = 2**attempt
                logger.warning(
                    f"Transient error on batch {start + 1}-{end} (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            except Exception as e:
                logger.error(
                    f"Unexpected error uploading batch {start + 1}-{end}: {e}",
                    exc_info=True,
                )
                # For unexpected errors, don't retry indefinitely; re-raise after logging.
                raise

    logger.info("Upload complete")


def main() -> None:
    load_env_variables()

    DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION_NAME = os.getenv("MONGO_TRACKS_COLLECTION")
    EMBEDDING_FEATURES = [
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    df = get_dataset(DATA_FILE_PATH)
    df = pre_process_data(df)
    all_embeddings = df[EMBEDDING_FEATURES].values

    client = None
    try:
        client, collection = connect_to_mongo(DB_NAME, COLLECTION_NAME)
        documents = prepare_documents(df, all_embeddings)
        empty_and_populate_collection(collection, documents)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    main()
