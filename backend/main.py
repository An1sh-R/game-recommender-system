from fastapi import FastAPI
from backend.routes.recommendation import router as recommendation_router

app = FastAPI()

# include recommendation routes
app.include_router(recommendation_router)


@app.get("/")
def home():
    return {"message": "Welcome to the Game Recommender API!"}