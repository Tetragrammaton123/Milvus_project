from .base import BaseAgent
from pydantic import BaseModel
from typing import List
from loguru import logger


class CombinatorResponse(BaseModel):
    queries: List[str]


class CombinatorAgent(BaseAgent):
    def generate_queries(self, concept: str, n: int = 3) -> CombinatorResponse:
        logger.info(f"[{self.name}] Generating {n} queries for concept: {concept}")
        prompt = (
            f"Generate {n} diverse, concise search queries for an academic search engine "
            f"covering the concept: {concept}."
        )
        try:
            result = self.llm.generate_structured(prompt, CombinatorResponse)
            logger.success(f"Generated {len(result.queries)} queries: {result.queries}")
            return result
        except Exception:
            logger.exception("Failed to generate queries via LLM. Falling back to default queries")
            parts = [concept, concept + " research", concept + " review"]
            fallback = CombinatorResponse(queries=parts[:n])
            logger.debug(f"Using fallback queries: {fallback.queries}")
            return fallback
