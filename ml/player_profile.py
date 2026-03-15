import numpy as np

# mapping quiz questions → traits
QUESTION_TRAITS = {
    0: "exploration",
    1: "story",
    2: "challenge",
    3: "strategy",
    4: "social",
    5: "relaxation",
    6: "exploration",
    7: "story",
    8: "strategy",
    9: "challenge"
}


TRAITS = [
    "exploration",
    "story",
    "challenge",
    "strategy",
    "social",
    "relaxation"
]


def compute_player_profile(answers): # answers -> list of 10 scores (1-5)
    trait_scores = {}
    for trait in TRAITS:
        trait_scores[trait] = []
    for index, answer in enumerate(answers):
        trait = QUESTION_TRAITS[index]
        trait_scores[trait].append(answer)
    # Average scores for each trait
    profile = {}
    for trait, scores in trait_scores.items():
        if scores:
            profile[trait] = float(np.mean(scores)/5) # normalize to 0-1
        else:
            profile[trait] = 0
    return profile

if __name__ == "__main__":
    # Example usage
    answers = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    profile = compute_player_profile(answers)
    print(profile)
    