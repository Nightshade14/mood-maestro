import logging
import pandas as pd
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Tuple, List
import os
from tqdm import tqdm
import time

from scripts.models import Entity


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


def get_entities(df: pd.DataFrame, embedding_features: list) -> Tuple[list, list, list]:
    albums_embeddings = (
        df.groupby("album_name")[embedding_features]
        .mean()
        .apply(list, axis=1)
        .reset_index(name="embedding")
        .rename(columns={"album_name": "name"})
        .to_dict("records")
    )

    track_genre_embeddings = (
        df.groupby("track_genre")[embedding_features]
        .mean()
        .apply(list, axis=1)
        .reset_index(name="embedding")
        .rename(columns={"track_genre": "name"})
        .to_dict("records")
    )

    df_exploded = df.assign(artists=df["artists"].str.split(";")).explode("artists")

    artists_embeddings = (
        df_exploded.groupby("artists")[embedding_features]
        .mean()
        .apply(list, axis=1)
        .reset_index(name="embedding")
        .rename(columns={"artists": "name"})
        .to_dict("records")
    )

    # embedding_features is a list of column names; the embedding length equals the number of features
    emb_len = len(embedding_features)
    logger.info(
        f"Extracted {len(albums_embeddings)} album embeddings of length {emb_len} each"
    )
    logger.info(
        f"Extracted {len(track_genre_embeddings)} track genre embeddings of length {emb_len} each"
    )
    logger.info(
        f"Extracted {len(artists_embeddings)} artist embeddings of length {emb_len} each"
    )

    return albums_embeddings, track_genre_embeddings, artists_embeddings


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


def prepare_documents(entity_embedding: list) -> List[dict]:
    documents: List[dict] = []
    for index in tqdm(
        range(len(entity_embedding)),
        total=len(entity_embedding),
        desc="Preparing Documents",
    ):
        emb_dict = entity_embedding[index]
        try:
            entity_document = Entity(
                entity_name=emb_dict["name"], embedding=emb_dict["embedding"]
            )
            documents.append(entity_document.model_dump())
        except Exception as e:
            logger.warning(
                f"Skipping entity {emb_dict.get('name')}: {e}", exc_info=True
            )
            continue
    return documents


def empty_and_populate_collection(
    collection: pymongo.collection.Collection, documents: list
) -> None:
    batch_size = 10000
    max_retries = 3
    logger.info(f"Uploading {len(documents)} documents in batches of {batch_size}...")

    logger.info("Emptying the collection...")
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
    DATASET_PATH = os.getenv("DATA_FILE_PATH")
    DB_NAME = os.getenv("DB_NAME")
    ALBUMS_COLLECTION_NAME = os.getenv("MONGO_ALBUMS_COLLECTION")
    GENRE_COLLECTION_NAME = os.getenv("MONGO_GENRES_COLLECTION")
    ARTISTS_COLLECTION_NAME = os.getenv("MONGO_ARTISTS_COLLECTION")
    EMBEDDING_FEATURES = os.getenv("EMBEDDING_FEATURES").split(",")

    df = get_dataset(DATASET_PATH)
    df = pre_process_data(df)
    albums_embeddings, track_genre_embeddings, artists_embeddings = get_entities(
        df, EMBEDDING_FEATURES
    )

    client = None
    try:
        client, collection = connect_to_mongo(DB_NAME, ALBUMS_COLLECTION_NAME)
        albums = prepare_documents(albums_embeddings)
        empty_and_populate_collection(collection, albums)

        client, collection = connect_to_mongo(DB_NAME, GENRE_COLLECTION_NAME)
        genres = prepare_documents(track_genre_embeddings)
        empty_and_populate_collection(collection, genres)

        client, collection = connect_to_mongo(DB_NAME, ARTISTS_COLLECTION_NAME)
        artists = prepare_documents(artists_embeddings)
        empty_and_populate_collection(collection, artists)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    main()
