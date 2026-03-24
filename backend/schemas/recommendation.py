from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class recommendationRequest(BaseModel):
    game: str
    user_id: Optional[int] = None
    quiz_answers: Optional[List[int]] = Field(default=None, min_length=10, max_length=10)
    top_n: int = 5

    @field_validator("quiz_answers")
    @classmethod
    def validate_quiz_answer_range(cls, values: Optional[List[int]]) -> Optional[List[int]]:
        if values is None:
            return values
        if any(value < 1 or value > 5 for value in values):
            raise ValueError("quiz_answers must contain values between 1 and 5")
        return values

class QuizRecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    quiz_answers: List[int] = Field(..., min_length=10, max_length=10)
    top_n: int = 5

    @field_validator("quiz_answers")
    @classmethod
    def validate_quiz_answer_range(cls, values: List[int]) -> List[int]:
        if any(value < 1 or value > 5 for value in values):
            raise ValueError("quiz_answers must contain values between 1 and 5")
        return values