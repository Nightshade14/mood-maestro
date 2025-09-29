import logging
import pandas as pd
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Tuple
import os
import time

from scripts.models import User


logger = logging.getLogger("mood_maestro.set_entities_collection")


def configure_logging_from_env() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_env_variables() -> None:
    load_dotenv(".env")
    configure_logging_from_env()
    logger.info("Environment loaded")


def get_dataset(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {file_path}")
    return pd.read_csv(file_path)


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=["track_id", "track_name"], inplace=True)
    df.drop_duplicates(subset=["track_id"], inplace=True)
    df = df.reset_index(drop=True)
    return df


def connect_to_mongo(
    db_name: str, collection_name: str
) -> Tuple[MongoClient, pymongo.collection.Collection]:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not found in environment variables.")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[db_name]
    collection = db[collection_name]
    logger.info("Connected to MongoDB")
    return client, collection


def prepare_document(user_embedding: list, name: str) -> dict:
    document: dict = {}
    try:
        user_document = User(
            name=name,
            genres_embedding=user_embedding,
            artists_embedding=user_embedding,
            tracks_embedding=user_embedding,
            albums_embedding=user_embedding,
        )
        document = user_document.model_dump()
    except Exception as e:
        logger.warning(f"Skipping user {name}: {e}", exc_info=True)
    return document


def empty_and_populate_collection(
    collection: pymongo.collection.Collection, document: dict
) -> None:
    batch_size = 10000
    max_retries = 3
    logger.info(f"Uploading {len(document)} documents in batches of {batch_size}...")

    logger.info("Emptying the collection...")
    collection.delete_many({})

    if not document:
        logger.info("No documents to upload.")
        return

    attempt = 0
    while attempt < max_retries:
        try:
            collection.insert_many([document], ordered=True)
            logger.info(f"Uploaded document successfully")
            break
        except pymongo.errors.BulkWriteError as bwe:
            # BulkWriteError can occur for duplicate _id or other write errors.
            # Log and continue if some writes succeeded.
            logger.warning(f"BulkWriteError on document: {bwe.details}")
            # If ordered=False some docs may have been written; stop retrying this batch.
            break
        except (pymongo.errors.AutoReconnect, pymongo.errors.NetworkTimeout) as e:
            attempt += 1
            wait = 2**attempt
            logger.warning(
                f"Transient error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s..."
            )
            time.sleep(wait)
        except Exception as e:
            logger.error(
                f"Unexpected error uploading document: {e}",
                exc_info=True,
            )
            # For unexpected errors, don't retry indefinitely; re-raise after logging.
            raise

    logger.info("Upload complete")


def main() -> None:
    load_env_variables()
    DATASET_PATH = os.getenv("DATA_FILE_PATH")
    DB_NAME = os.getenv("DB_NAME")
    USERS_COLLECTION_NAME = os.getenv("MONGO_USERS_COLLECTION")
    EMBEDDING_FEATURES = os.getenv("EMBEDDING_FEATURES").split(",")

    df = get_dataset(DATASET_PATH)
    df = pre_process_data(df)
    embedding = df[EMBEDDING_FEATURES].mean().tolist()

    client = None
    try:
        client, collection = connect_to_mongo(DB_NAME, USERS_COLLECTION_NAME)
        user = prepare_document(embedding, "default_user")
        empty_and_populate_collection(collection, user)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    main()
