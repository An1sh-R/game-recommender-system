import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed/steam_games_cleaned.csv")
print("Games Loaded:",len(df))
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words = "english",max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# save the vectorizer and the matrix
with open("data/vectors/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("data/vectors/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

# Select only what the user needs to see in the recommender UI
cols_to_keep = ['AppID', 'Name', 'Genres', 'Tags','About the game',"popularity"]
metadata_df = df[cols_to_keep]

# Save the slimmed-down version
metadata_df.to_csv("data/vectors/games_metadata.csv", index=False)
print("Vectors and metadata saved successfully.")
