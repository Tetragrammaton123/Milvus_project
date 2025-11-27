from .base import BaseAgent, SearchToolParams, CombinatorToolParams
from pydantic import BaseModel
from typing import Literal, Optional
from loguru import logger


class IntentResponse(BaseModel):
    intent: Literal["find_one", "build_set", "out-of-scope"]


class ChatAgent(BaseAgent):
    def _format_history_context(self, conversation_history: list[dict]) -> str:
        """Format conversation history into a context string for the LLM."""
        if not conversation_history:
            return ""
        
        # Take last 3 exchanges to keep context manageable
        recent_history = conversation_history[-3:]
        context_parts = []
        
        for exchange in recent_history:
            user_msg = exchange.get("user", "")
            assistant_msg = exchange.get("assistant", "")
            
            context_parts.append(f"User: {user_msg}")
            if assistant_msg:
                context_parts.append(f"Assistant: {assistant_msg}")
        
        return "\n".join(context_parts)

    def classify_intent(self, query: str, conversation_history: Optional[list[dict]] = None) -> IntentResponse:
        logger.debug(f"Classifying intent for query: {query}")
        
        # Build prompt with optional conversation context
        prompt_parts = []
        
        if conversation_history:
            history_context = self._format_history_context(conversation_history)
            if history_context:
                prompt_parts.append("Previous conversation context:")
                prompt_parts.append(history_context)
                prompt_parts.append("")  # Empty line for separation
        
        prompt_parts.extend([
            "Classify the intent of the user's query in the context of academic literature search.",
            "",
            f"Current query: {query}",
            "",
            "Possible intents:",
            "- find_one (find one specific academic paper)",
            "- build_set (create a set or collection of academic papers)",
            "- out-of-scope (query is not about academic literature)"
        ])
        
        prompt = "\n".join(prompt_parts)
        result = self.llm.generate_structured(prompt, IntentResponse)
        logger.debug(f"Intent classified as: {result.intent}")
        return result

    def act(self, user_input: str, conversation_history: Optional[list[dict]] = None):
        logger.info(f"[{self.name}] received input: {user_input}")
        
        # Pass history to intent classification
        intent_resp = self.classify_intent(user_input, conversation_history)
        logger.info(f"Intent classified: {intent_resp.intent}")
        
        if intent_resp.intent == "out-of-scope":
            logger.debug("Query is out-of-scope, returning early")
            print(" Your query does not appear to be about academic research. Please ask for scientific papers or studies.")
            return {"intent": "out-of-scope", "message": "Please ask for scientific papers or studies."}

        # Build prompt with conversation context
        prompt_parts = []
        if conversation_history:
            history_context = self._format_history_context(conversation_history)
            if history_context:
                prompt_parts.append("Previous conversation:")
                prompt_parts.append(history_context)
                prompt_parts.append("")

        if intent_resp.intent in ["find_one", "build_set"]:
            tool_model = SearchToolParams
            action_name = "search"
            prompt_parts.append(f"Given the user query: '{user_input}', generate structured SearchToolParams JSON output.")
        else:
            tool_model = CombinatorToolParams
            action_name = "generate_queries"
            prompt_parts.append(f"Given the user query: '{user_input}', generate structured CombinatorToolParams JSON output.")

        llm_prompt = "\n".join(prompt_parts)
        logger.debug(f"Generating structured parameters for action: {action_name}")
        tool_params = self.llm.generate_structured(llm_prompt, tool_model)
        logger.debug(f"Structured tool params: {tool_params}")

        if action_name in self.tools:
            logger.info(f"Executing tool: {action_name}")
            tool_func = self.tools[action_name]
            result = tool_func(**tool_params.model_dump(exclude="reasoning"))
            logger.success(f"Tool execution completed: {action_name}")
            logger.debug(f"Tool result: {result}")
            return result
        
        logger.error(f"Unknown action: {action_name}")
        return None
