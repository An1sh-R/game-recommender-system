from fastapi import APIRouter
from backend.schemas.recommendation import recommendationRequest
from backend.schemas.recommendation import QuizRecommendationRequest
from backend.services.recommendation_service import get_recommendations
from backend.services.recommendation_service import get_quiz_recommendations

router = APIRouter()


@router.post("/recommend/game")
def recommend_game(req: recommendationRequest):

    recommendation = get_recommendations(
        game=req.game,
        quiz_answers=req.quiz_answers,
        top_n=req.top_n
    )

    return {
        "mode": "game",
        "query": req.game,
        "recommendations": recommendation
    }
@router.post("/recommend/quiz")
def recommend_quiz(req: QuizRecommendationRequest):

    recommendations = get_quiz_recommendations(
        quiz_answers = req.quiz_answers,
        top_n = req.top_n
    )

    return {
        "mode" : "quiz",
        "recommendations": recommendations
    }