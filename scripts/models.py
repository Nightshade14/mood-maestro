from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TrackAudioFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float


class TrackDocument(BaseModel):
    id: str = Field(..., alias="_id")
    track_name: str
    artists: List[str]
    album_name: Optional[str]
    track_genre: Optional[str]
    popularity: float
    duration_ms: float
    explicit: bool
    audio_features: TrackAudioFeatures
    embedding: List[float]
    skip_count: int
    finish_count: int
    liked: bool
    cooldown_level: int
    last_skip_timestamp: int | None

    @field_validator("embedding")
    def embedding_must_not_be_empty(cls, v: List[float]):
        if not v or len(v) == 0:
            raise ValueError("embedding must be a non-empty list of floats")
        return v

    class Config:
        # Pydantic v2 config keys
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "123",
                "track_name": "Song",
                "artists": ["Artist 1", "Artist 2"],
                "album_name": "Album",
                "track_genre": "Pop",
                "popularity": 0.5,
                "duration_ms": 210000,
                "explicit": False,
                "audio_features": {
                    "danceability": 0.5,
                    "energy": 0.6,
                    "key": 5,
                    "loudness": -5.2,
                    "mode": 1,
                    "speechiness": 0.03,
                    "acousticness": 0.02,
                    "instrumentalness": 0.0,
                    "liveness": 0.12,
                    "valence": 0.4,
                    "tempo": 120.0,
                },
                "embedding": [0.1, 0.2, 0.3],
            }
        }
