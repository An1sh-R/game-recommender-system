from pydantic import BaseModel
from typing import List

class recommendationRequest(BaseModel):
    game : str
    quiz_answers : List[int]
    top_n : int = 5
