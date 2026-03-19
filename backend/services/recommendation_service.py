import json
import pickle
import redis
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from ml.content_recommender import recommend_games
from ml.player_profile import compute_player_profile


# Redis connection
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)


# load game trait vectors once
with open("data/vectors/game_traits.pkl", "rb") as f:
    game_traits = pickle.load(f)


TRAIT_ORDER = [
    "exploration",
    "story",
    "challenge",
    "strategy",
    "social",
    "relaxation"
]


def profile_to_vector(profile: Dict[str, float]) -> np.ndarray:
    return np.array([profile[t] for t in TRAIT_ORDER])


def trait_similarity(player_vec: np.ndarray, game_vec: np.ndarray) -> float:
    return cosine_similarity(
        player_vec.reshape(1, -1),
        game_vec.reshape(1, -1)
    )[0][0]


def get_recommendations(game: str, quiz_answers: List[int], top_n: int = 5) -> List[Dict]:

    cache_key = f"recommendations:{game}:{top_n}:{','.join(map(str, quiz_answers))}"

    cached_result = redis_client.get(cache_key)

    if cached_result is not None:
        print("Cache hit")
        return json.loads(str(cached_result))

    print("Cache miss")

    # Tell Pylance what this is
    results: List[Dict] = recommend_games(game, top_n=50)

    # compute player profile
    player_profile = compute_player_profile(quiz_answers)
    player_vec = profile_to_vector(player_profile)

    for r in results:

        idx = r["index"]   # index must come from content_recommender

        game_vec = np.array(list(game_traits[idx].values()))

        player_match = trait_similarity(player_vec, game_vec)

        r["player_match"] = float(player_match)

        r["hybrid_score"] = (
            0.6 * r["similarity_score"]
            + 0.2 * r["popularity"]
            + 0.2 * r["player_match"]
        )

    results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

    results = results[:top_n]

    redis_client.set(cache_key, json.dumps(results), ex=3600)

    return results



GAME_METADATA = pd.read_csv("data/vectors/games_metadata.csv")
game_trait_matrix = np.array([list(trait_dict.values()) for trait_dict in game_traits])

def get_quiz_recommendations(quiz_answers: List[int], top_n: int = 5,metadata=GAME_METADATA) -> List[Dict]:
    cache_key = f"quiz_recommendation:{top_n}:{','.join(map(str,quiz_answers))}"
    cached_result = redis_client.get(cache_key)

    if cached_result is not None:
        print("Quiz Cache hit")
        return json.loads(str(cached_result))

    print("Quiz Cache miss")
    player_profile = compute_player_profile(quiz_answers)
    player_vec = profile_to_vector(player_profile)
    results = []

    # VECTORIZED similarity (FAST)
    similarities = cosine_similarity(
        player_vec.reshape(1, -1),
        game_trait_matrix
    ).flatten().astype(float)
    popularity = np.array(metadata["popularity"], dtype=float)
    # hybrid score
    hybrid_scores = 0.7 * similarities + 0.3 * popularity
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        game_info = metadata.iloc[idx]

        results.append({
            "Name": game_info["Name"],
            "Genres": game_info["Genres"],
            "Tags": game_info["Tags"],
            "popularity": float(popularity[idx]),
            "player_match": float(similarities[idx]),
            "hybrid_score": float(hybrid_scores[idx])
        })
    redis_client.set(cache_key, json.dumps(results), ex=3600)
    return results
 