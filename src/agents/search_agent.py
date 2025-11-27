from typing import List, Dict
from .base import BaseAgent
from pydantic import BaseModel
from typing import Any
from loguru import logger

class RewriteResponse(BaseModel):
    query: str


class SearchAgent(BaseAgent):
    embedding_model: Any = None
    collection: Any = None

    def rewrite_query(self, query: str) -> RewriteResponse:
        logger.debug(f"Rewriting query for vector search: {query}")
        prompt = (
            f"Rewrite the following query to be suitable for a scientific article vector search.\n"
            f"User query: {query}"
        )
        result = self.llm.generate_structured(prompt, RewriteResponse)
        logger.debug(f"Query rewritten to: {result.query}")
        return result

    def search_vector_index(self, query: str, limit: int = 5) -> List[Dict]:
        if self.embedding_model is None or self.collection is None:
            logger.error("SearchAgent not configured with embedding_model and collection")
            raise RuntimeError("SearchAgent not configured with embedding_model and collection")

        logger.info(f"[{self.name}] Searching for: {query}")
        logger.debug("Encoding query into embedding vector")
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        
        logger.debug(f"Searching collection with limit={limit}")
        results = self.collection.search(
            query_emb,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["title", "abstract"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "title": hit.entity.get("title"),
                "abstract": hit.entity.get("abstract"),
                "score": float(hit.distance)
            })

        logger.success(f"[{self.name}] Found {len(hits)} results")
        return hits

    def search(self, query: str, limit: int = 5) -> dict:
        logger.info(f"Starting search for query: {query}")
        rewritten = self.rewrite_query(query)
        results = self.search_vector_index(rewritten.query, limit=limit)
        logger.success(f"Search completed with {len(results)} results")
        return {"rewritten": rewritten.query, "results": results}
