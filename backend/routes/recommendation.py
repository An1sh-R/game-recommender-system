from fastapi import APIRouter
from backend.schemas.recommendation import recommendationRequest
from backend.schemas.recommendation import QuizRecommendationRequest
from backend.services.recommendation_service import get_recommendations
from backend.services.recommendation_service import get_quiz_recommendations
from backend.services.user_service import save_user_profile
from ml.player_profile import compute_player_profile

router = APIRouter()


@router.post("/recommend/game")
def recommend_game(req: recommendationRequest):

    result = get_recommendations(
        game=req.game,
        quiz_answers=req.quiz_answers,
        top_n=req.top_n,
        user_id=req.user_id,
    )

    return {
        "mode": result["mode"],
        "query": req.game,
        "recommendations": result["recommendations"],
    }
@router.post("/recommend/quiz")
def recommend_quiz(req: QuizRecommendationRequest):
    # Quiz submission is stateful: compute profile and save it for future sessions.
    profile = compute_player_profile(req.quiz_answers)
    save_user_profile(req.user_id, profile)

    result = get_quiz_recommendations(
        quiz_answers=req.quiz_answers,
        top_n=req.top_n,
        user_id=req.user_id,
    )

    return {
        "mode": result["mode"],
        "user_id": req.user_id,
        "recommendations": result["recommendations"],
    }