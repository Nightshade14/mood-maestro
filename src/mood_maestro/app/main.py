from fastapi import FastAPI

from . import schemas, services

app = FastAPI(title="Mood Maestro API")


@app.post("/recommendations", response_model=schemas.RecommendationResponse)
def create_recommendations(request: schemas.RecommendationRequest):
    """
    Endpoint to get music recommendations based on a natural language query.
    """
    # The endpoint's only job is to delegate to the service layer
    return services.get_song_recommendations(request.query, request.user_id)
