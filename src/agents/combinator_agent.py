from .base import BaseAgent
from pydantic import BaseModel
from typing import List


class CombinatorResponse(BaseModel):
    queries: List[str]


class CombinatorAgent(BaseAgent):
    def generate_queries(self, concept: str, n: int = 3) -> CombinatorResponse:
        prompt = (
            f"Generate {n} diverse, concise search queries for an academic search engine "
            f"covering the concept: {concept}."
        )
        try:
            return self.llm.generate_structured(prompt, CombinatorResponse)
        except Exception:
            parts = [concept, concept + " research", concept + " review"]
            return CombinatorResponse(queries=parts[:n])
