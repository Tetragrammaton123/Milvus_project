import re
from typing import Any, Type, TypeVar, Optional
from pydantic import BaseModel
from openai import OpenAI
from pydantic import BaseModel, Field

class SearchToolParams(BaseModel):
    reasoning: str = Field(description="Preliminary speculations on how to generate proper search query")
    query: str = Field(description="Search query to address to search agent with")
    limit: int = Field(default=5, description="Number of search results to return")

class CombinatorToolParams(BaseModel):
    reasoning: str = Field(description="Preliminary speculations on how to expand a concept into multiple queries")
    concept: str = Field(description="The concept for which to generate diverse search queries")
    n: int = Field(default=3, description="Number of queries to generate")
    
T = TypeVar("T", bound=BaseModel)
JSON_EXTRACT_RE = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])", re.DOTALL)


class OpenAILLM:
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_system = "You are a reasoning agent. Always respond in JSON when asked."

    def generate_structured(self, prompt: str, response_model: Type[T]) -> Optional[T]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.base_system},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
        )
        content = (
            completion.choices[0].message.parsed
            if hasattr(completion.choices[0].message, "parsed")
            else completion.choices[0].message.content
        )
        return response_model.model_validate_json(content)


class BaseAgent(BaseModel):
    name: str
    role: str
    llm: Any
    tools: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

    def add_tool(self, name: str, func: Any):
        self.tools[name] = func
