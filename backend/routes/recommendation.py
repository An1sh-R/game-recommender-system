from fastapi import APIRouter
from backend.schemas.recommendation import recommendationRequest
from backend.services.recommendation_service import get_recommendations

router = APIRouter()


@router.post("/recommend")
def recommend(req: recommendationRequest):

    recommendations = get_recommendations(
        req.game,
        req.quiz_answers,
        req.top_n
    )

    return {
        "query": req.game,
        "recommendations": recommendations
    }