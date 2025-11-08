from .base import BaseAgent
from pydantic import BaseModel
from typing import Literal, Dict


class IntentResponse(BaseModel):
    intent: Literal["find_one", "build_set", "out-of-scope"]


class Thought(BaseModel):
    reasoning: str
    next_action: str
    args: dict


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

    def decide_next(self, user_input: str) -> Thought:
        prompt = (
            "You are a reasoning agent coordinating tasks.\n"
            f"User said: {user_input}\n\n"
            "Possible actions: classify_intent, search, generate_queries, or finish.\n"
            "If you call search, you must provide args as {\"query\": \"...\"}.\n"
            "If you call generate_queries, provide args as {\"concept\": \"...\"}."
        )
        try:
            return self.llm.generate_structured(prompt, Thought)
        except Exception:
            return Thought(reasoning="fallback", next_action="classify_intent", args={"query": user_input})

    def act(self, user_input: str):
        print(f"\n [{self.name}] received input: {user_input}")
        intent_resp = self.classify_intent(user_input)
        print(f"Intent classified: {intent_resp.intent}")
        if intent_resp.intent == "out-of-scope":
            print(" Your query does not appear to be about academic research. Please ask for scientific papers or studies.")
            return {"intent": "out-of-scope", "message": "Please ask for scientific papers or studies."}

        thought = self.decide_next(user_input)
        print(f"Reasoning: {thought.reasoning}")
        print(f"Next action: {thought.next_action}")

        action = thought.next_action
        args = thought.args or {}

        if "topic" in args and "query" not in args:
            args["query"] = args.pop("topic")
        if "question" in args and "query" not in args:
            args["query"] = args.pop("question")
        if "subject" in args and "query" not in args:
            args["query"] = args.pop("subject")

        query = args.get("query", user_input)
        args["query"] = query

        if action == "search":
            intent_resp = self.classify_intent(args.get("query", user_input))
            if intent_resp.intent == "find_one":
                args["limit"] = 1
            elif intent_resp.intent == "build_set":
                args["limit"] = args.get("limit", 5)

        if action == "finish":
            print("Conversation complete.")
            return None

        if action in self.tools:
            tool = self.tools[action]
            result = tool(**args)
            print(f" Tool result: {result}")
            return result

        print(f" Unknown action: {action}")
        return None
