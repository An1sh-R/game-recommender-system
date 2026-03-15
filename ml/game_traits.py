import pandas as pd
import pickle


TRAITS = [
    "exploration",
    "story",
    "challenge",
    "strategy",
    "social",
    "relaxation"
]


TAG_TO_TRAIT = {

    "Adventure": ["exploration", "story"],
    "RPG": ["story", "exploration"],
    "Action": ["challenge"],
    "Strategy": ["strategy"],
    "Simulation": ["relaxation", "strategy"],
    "Casual": ["relaxation"],
    "Indie": ["exploration"],
    "Sports": ["challenge", "social"],
    "Racing": ["challenge"],
    "Multiplayer": ["social"],
}


def compute_game_traits(genres, tags):

    scores = {trait: 0.0 for trait in TRAITS}

    combined = f"{genres},{tags}"

    tag_list = [t.strip() for t in combined.split(",") if t.strip()] 

    for tag in tag_list:

        if tag in TAG_TO_TRAIT:

            for trait in TAG_TO_TRAIT[tag]:
                scores[trait] += 1.0

    # normalize using saturation point
    SATURATION_POINT = 3.0

    for trait in scores:
        scores[trait] = float(min(1.0, scores[trait] / SATURATION_POINT))

    return scores


if __name__ == "__main__":

    df = pd.read_csv("data/processed/steam_games_cleaned.csv")

    print("Computing trait vectors...")

    trait_vectors = []

    for _, row in df.iterrows():

        traits = compute_game_traits(
            row["Genres"],
            row["Tags"]
        )

        trait_vectors.append(traits)

    print("Saving trait vectors...")

    with open("data/vectors/game_traits.pkl", "wb") as f:
        pickle.dump(trait_vectors, f)

    print("Done.")