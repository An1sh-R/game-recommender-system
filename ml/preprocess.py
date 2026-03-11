import pandas as pd

df = pd.read_csv('data/raw/games.csv')
print("Original Shape:", df.shape)

# select useful columns
df = df[["AppID","Name","Genres","Categories","Tags","About the game","Positive","Negative"]]
print("After selecting colums:", df.shape)
df.dropna(inplace=True)
print("After dropping NaN values:", df.shape)

# feature engineering
df["popularity"] = df["Positive"] / (df["Positive"] + df["Negative"])
df["combined_text"] = df["Genres"].astype(str) + " " + df["Categories"].astype(str) + " " + df["Tags"].astype(str) + " " + df["About the game"].astype(str)
df["combined_text"] = df["combined_text"].str.lower() # helps to treat tf-idf words consistently
print(df[["Name", "combined_text"]].head())

# save cleaned dataset
df.to_csv("data/processed/steam_games_cleaned.csv", index=False)

print("Cleaned dataset saved.")