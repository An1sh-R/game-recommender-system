import json
import redis
from ml.content_recommender import recommend_games

# connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def get_recommendations(game: str, top_n: int = 5):

    # check if recommendations are cached in Redis
    cache_key = f"recommendations:{game}:{top_n}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        # This assertion makes Pylance happy
        assert isinstance(cached_result, str)
        print("Cache hit")
        return json.loads(cached_result)
    
    print("Cache miss")
    results = recommend_games(game, top_n)
    # cache the results in Redis for 1 hour
    redis_client.set(cache_key, json.dumps(results), ex=3600)
    return results