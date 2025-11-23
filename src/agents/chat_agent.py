from .base import BaseAgent, SearchToolParams, CombinatorToolParams
from pydantic import BaseModel
from typing import Literal, Dict


class IntentResponse(BaseModel):
    intent: Literal["find_one", "build_set", "out-of-scope"]


class ChatAgent(BaseAgent):
    def classify_intent(self, query: str) -> IntentResponse:
        prompt = (
            "Classify the intent of the user's query in the context of academic literature search.\n\n"
            f"Query: {query}\n\n"
            "Possible intents:\n"
            "- find_one (find one specific academic paper)\n"
            "- build_set (create a set or collection of academic papers)\n"
            "- out-of-scope (query is not about academic literature)"
        )
        return self.llm.generate_structured(prompt, IntentResponse)

    def act(self, user_input: str):
        print(f"\n [{self.name}] received input: {user_input}")
        intent_resp = self.classify_intent(user_input)
        print(f"Intent classified: {intent_resp.intent}")
        if intent_resp.intent == "out-of-scope":
            print(" Your query does not appear to be about academic research. Please ask for scientific papers or studies.")
            return {"intent": "out-of-scope", "message": "Please ask for scientific papers or studies."}

        if intent_resp.intent in ["find_one", "build_set"]:
            tool_model = SearchToolParams
            action_name = "search"
            llm_prompt = f"Given the user query: '{user_input}', generate structured SearchToolParams JSON output."
        else:
            tool_model = CombinatorToolParams
            action_name = "generate_queries"
            llm_prompt = f"Given the user query: '{user_input}', generate structured CombinatorToolParams JSON output."

        tool_params = self.llm.generate_structured(llm_prompt, tool_model)
        print(f"Structured tool params: {tool_params}")

        if action_name in self.tools:
            tool_func = self.tools[action_name]
            result = tool_func(**tool_params.dict())
            print(f"Tool result: {result}")
            return result
        print(f"Unknown action: {action_name}")
        return None
