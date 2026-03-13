import difflib
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the tf-idf vectors
with open("data/vectors/tfidf_matrix.pkl","rb") as f:
    tfidf_matrix = pickle.load(f)
# load metadata
metadata = pd.read_csv("data/vectors/games_metadata.csv")

name_to_idx = {name.lower(): idx for idx, name in enumerate(metadata["Name"])} # mapping for quick lookup

def recommend_games(game_name, top_n=5):
    game_name = game_name.lower()
    # 1 Exact match
    if game_name in name_to_idx:
        idx = name_to_idx[game_name]
    else:
        # 2 substring match
        substring_matches = metadata[metadata["Name"].str.lower().str.contains(game_name, na=False)]
        if not substring_matches.empty:
            idx = substring_matches.index[0]
            print(f"Using substring match: {metadata.loc[idx,'Name']}")
        else:
            # fuzzy match fallback
            close_matches = difflib.get_close_matches(game_name, name_to_idx.keys(), n=1, cutoff=0.8)
            if not close_matches:
                print("Game not found")
                return
            matched_name = close_matches[0]
            print(f"Using closest match: {matched_name}")
            idx = name_to_idx[matched_name]

    # compute cosine similarity
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similarities = (similarities-similarities.min()) / (similarities.max() - similarities.min() + 1e-10) # added 1e-10 to prevent division by zero
    # combine similarity with popularity
    popularity_scores = metadata["popularity"].fillna(0).values
    weighted_scores = (0.8 * similarities + 0.2 * popularity_scores)
    # sort by weighted score
    similar_indices = weighted_scores.argsort()[::-1][1:top_n+1]
    results = metadata.iloc[similar_indices][["Name", "Genres", "Tags", "popularity"]].copy() 
    results["similarity_score"] = similarities[similar_indices] 
    results["weighted_score"] = weighted_scores[similar_indices]
    return results
if __name__ == "__main__":
    print(recommend_games("witcher 3", 5))