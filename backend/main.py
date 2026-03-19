from fastapi import FastAPI
from backend.routes.recommendation import router as recommendation_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# include recommendation routes
app.include_router(recommendation_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the Game Recommender API!"}