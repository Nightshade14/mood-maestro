import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Tuple
import logging

from scripts.models import TrackAudioFeatures, TrackDocument


# Module logger
logger = logging.getLogger("mood_maestro.set_db")


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
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["track_id"], inplace=True)
    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame, numerical_features: list) -> pd.DataFrame:
    logger.info("Performing feature engineering...")
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    df["text_features"] = (
        df["track_name"]
        + " by "
        + df["artists"]
        + " is part of the album "
        + df["album_name"]
        + " in the genre "
        + df["track_genre"]
    )
    logger.info("Feature engineering complete.")
    return df


def embed_features(
    df: pd.DataFrame, numerical_features: list, embedding_model_name: str
) -> np.ndarray:
    logger.info(f"Generating embeddings using '{embedding_model_name}'...")
    model = SentenceTransformer(embedding_model_name)
    text_embeddings = model.encode(df["text_features"].tolist(), show_progress_bar=True)
    numerical_data = df[numerical_features].to_numpy()
    combined_embeddings = np.hstack((text_embeddings, numerical_data))
    logger.info(
        f"Embeddings generated. Vector dimension: {combined_embeddings.shape[1]}"
    )
    return combined_embeddings


def load_env_variables() -> None:
    load_dotenv(".env")
    configure_logging_from_env()
    logger.info("Environment loaded")


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


def prepare_documents(df: pd.DataFrame, combined_embeddings: np.ndarray) -> List[dict]:
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
                duration_ms=float(row["duration_ms"]),
                explicit=bool(row["explicit"]),
                audio_features=audio,
                embedding=combined_embeddings[index].tolist(),
            )
            documents.append(doc_model.model_dump(by_alias=True))
        except Exception as e:
            logger.warning(f"Skipping row {index}: {e}", exc_info=True)
            continue
    return documents


def empty_and_populate_collection(
    collection: pymongo.collection.Collection, documents: list
) -> None:
    logger.info(f"Uploading {len(documents)} documents...")
    collection.delete_many({})
    if documents:
        collection.insert_many(documents, ordered=True)
    logger.info("Upload complete")


def main() -> None:
    load_env_variables()

    DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    NUMERICAL_FEATURES = [
        "popularity",
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
    df = engineer_features(df, NUMERICAL_FEATURES)
    combined_embeddings = embed_features(df, NUMERICAL_FEATURES, EMBEDDING_MODEL_NAME)

    client = None
    try:
        client, collection = connect_to_mongo(DB_NAME, COLLECTION_NAME)
        documents = prepare_documents(df, combined_embeddings)
        empty_and_populate_collection(collection, documents)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    main()
