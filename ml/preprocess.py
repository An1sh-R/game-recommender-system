import pandas as pd

df = pd.read_csv('data/raw/games.csv')
print(df.head())
print(df.columns)
print(len(df))