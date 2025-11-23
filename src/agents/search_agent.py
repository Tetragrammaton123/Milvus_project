from typing import List, Dict
from .base import BaseAgent, SearchToolParams, CombinatorToolParams
from pydantic import BaseModel
from typing import Any

class RewriteResponse(BaseModel):
    query: str


class SearchAgent(BaseAgent):
    embedding_model: Any = None
    collection: Any = None

    def rewrite_query(self, query: str) -> RewriteResponse:
        prompt = (
            f"Rewrite the following query to be suitable for a scientific article vector search.\n"
            f"User query: {query}"
        )
        return self.llm.generate_structured(prompt, RewriteResponse)

    def search_vector_index(self, query: str, limit: int = 5) -> List[Dict]:
        if self.embedding_model is None or self.collection is None:
            raise RuntimeError("SearchAgent not configured with embedding_model and collection")

        print(f"[search_agent] Searching for: {query}")
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        results = self.collection.search(
            query_emb,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["title"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "title": hit.entity.get("title"),
                "score": float(hit.distance)
            })

        print(f"[search_agent] Found {len(hits)} results.")
        return hits

    def search(self, query: str, limit: int = 5) -> dict:
        rewritten = self.rewrite_query(query)
        results = self.search_vector_index(rewritten.query, limit=limit)
        return {"rewritten": rewritten.query, "results": results}
