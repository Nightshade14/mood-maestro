from ..agents.agent_manager import run_recommendation_flow
from .schemas import RecommendationResponse, TrackResponse


def get_song_recommendations(query: str) -> RecommendationResponse:
    """
    Service layer function to get recommendations.
    """
    # Call the agent manager to run the complex AI logic
    results = run_recommendation_flow(query)

    # Format the raw agent output into a clean API response
    response_tracks = [TrackResponse(**track) for track in results]
    return RecommendationResponse(recommendations=response_tracks)
