import pandas as pd
import numpy as np
df = pd.read_csv('data/raw/games.csv',index_col=False) # index_col=False to prevent first column from being treated as index (nightmare)
print("Original Shape:", df.shape)

# select useful columns
df = df[["AppID","Name","Genres","Categories","Tags","About the game","Estimated owners"]]
print("After selecting colums:", df.shape)
df.dropna(inplace=True)
print("After dropping NaN values:", df.shape)

# feature engineering
def parse_owners(val):
    try:
        low, high = val.split("-")
        return (int(low.strip()) + int(high.strip())) / 2
    except:
        return 0

df["owners_mid"] = df["Estimated owners"].apply(parse_owners)

df["popularity"] = np.log(df["owners_mid"] + 1) # add 1 to avoid log(0)
df["popularity"] = df["popularity"] / df["popularity"].max() # scale to 0–1
df["combined_text"] = df["Genres"].astype(str) + " " + df["Categories"].astype(str) + " " + df["Tags"].astype(str) + " " + df["About the game"].astype(str)
df["combined_text"] = df["combined_text"].str.lower() # helps to treat tf-idf words consistently
print(df[["Name", "combined_text"]].head())

# save cleaned dataset
df.to_csv("data/processed/steam_games_cleaned.csv", index=False)
print("Cleaned dataset saved.")