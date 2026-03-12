import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the tf-idf vectors
with open("data/vectors/tfidf_matrix.pkl","rb") as f:
    tfidf_matrix = pickle.load(f)
# load metadata
metadata = pd.read_csv("data/vectors/games_metadata.csv")

def recommend_games(game_name, top_n=5):
    # find idx of the game
    matches = metadata[metadata["Name"].str.lower().str.contains(game_name.lower())]
    if(matches.empty):
        print(f"Game '{game_name}' not found in the dataset.")
        return
    idx = matches.index[0]
    # compute cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix[idx],tfidf_matrix).flatten()
    cos_sim_idx = cos_sim.argsort()[::-1][1:top_n+1]  # get top n similar games, excluding itself
    return metadata.iloc[cos_sim_idx][["Name","Genres","Tags","popularity"]]
if __name__ == "__main__":
    print(recommend_games("The Witcher 3", 5))