from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    query: str


class TrackResponse(BaseModel):
    track_id: str
    track_name: str
    score: float


class RecommendationResponse(BaseModel):
    recommendations: list[TrackResponse]
