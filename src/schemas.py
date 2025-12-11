from pydantic import BaseModel
from typing import List


# --- Input Models ---
class ArticleInput(BaseModel):
    title: str
    content: str

class UserQuery(BaseModel):
    text: str
    top_n: int = 3

# --- Output Models ---

class ClassificationResponse(BaseModel):
    category: str
    confidence_score: float

class RecommendationItem(BaseModel):
    title: str
    category: str
    similarity_score: float

class RecommendationResponse(BaseModel):
    results: List[RecommendationItem]

