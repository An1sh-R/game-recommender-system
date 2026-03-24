from typing import Dict, List


# Each of the 10 quiz questions contributes to one or more traits.
# Weights are additive and let "hybrid" questions shape multiple dimensions.
QUESTION_TO_TRAITS = {
    0: {"exploration": 1.0},
    1: {"story": 1.0},
    2: {"challenge": 1.0},
    3: {"strategy": 1.0},
    4: {"social": 1.0},
    5: {"relaxation": 1.0},
    6: {"exploration": 0.6, "relaxation": 0.4},
    7: {"story": 0.6, "social": 0.4},
    8: {"strategy": 0.6, "challenge": 0.4},
    9: {"challenge": 0.5, "exploration": 0.5},
}


TRAITS = [
    "exploration",
    "story",
    "challenge",
    "strategy",
    "social",
    "relaxation"
]


def _normalize_answer(answer: int) -> float:
    # Normalize from [1..5] to [0..1]
    return (float(answer) - 1.0) / 4.0


def compute_player_profile(answers: List[int]) -> Dict[str, float]:
    """
    Build a trait profile from 10 quiz answers.

    Steps:
    1) Normalize each answer from [1..5] to [0..1]
    2) Add weighted contribution(s) to trait totals
    3) Normalize by total weight per trait to keep final scores in [0, 1]
    """
    if len(answers) != 10:
        raise ValueError("Expected exactly 10 quiz answers")
    if any(answer < 1 or answer > 5 for answer in answers):
        raise ValueError("Quiz answers must be integers between 1 and 5")

    trait_totals = {trait: 0.0 for trait in TRAITS}
    trait_weight_sums = {trait: 0.0 for trait in TRAITS}

    for question_index, answer in enumerate(answers):
        normalized_answer = _normalize_answer(answer)
        mapping = QUESTION_TO_TRAITS[question_index]

        for trait, weight in mapping.items():
            trait_totals[trait] += normalized_answer * weight
            trait_weight_sums[trait] += weight

    profile = {}
    for trait in TRAITS:
        total_weight = trait_weight_sums[trait]
        if total_weight == 0:
            profile[trait] = 0.0
        else:
            # Weighted mean naturally remains within [0, 1].
            profile[trait] = float(trait_totals[trait] / total_weight)

    return profile

if __name__ == "__main__":
    # Example usage
    answers = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]
    profile = compute_player_profile(answers)
    print(profile)
    