from fastapi import FastAPI
from backend.services.recommendation_service import get_recommendations
app = FastAPI()

@app.get("/")
def home():
    return {"Message": "Welcome to the Game Recommender API!"}

@app.get("/recommend")
def recommend(game: str, top_n: int = 5):
    recommendations = get_recommendations(game, top_n)
    return {"Query": game, "Recommendations": recommendations}