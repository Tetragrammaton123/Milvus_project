from .base import BaseAgent, SearchToolParams, CombinatorToolParams
from pydantic import BaseModel
from typing import Literal, Optional
from loguru import logger
from typing import Any
from uuid import UUID
from src.tool_suggest_intents import INTENT_TOOL_IDS


class IntentResponse(BaseModel):
    intent: Literal["find_one", "build_set", "out-of-scope"]


class ChatAgent(BaseAgent):
    ts_client: Any = None
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

    async def classify_intent(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None,
        *,
        parent_sample_id=None,
    ) -> IntentResponse:
        logger.debug(f"Classifying intent for query: {query}")

        ts_messages = None
        if self.ts_client is not None:
            try:
                ts_messages = self._to_ts_messages(query, conversation_history)
            except Exception:
                logger.exception("Failed to build ToolSuggest context messages")
                ts_messages = None

        # 1) Если ToolSuggest обучен — берём top-1 suggestion как интент
        if self.ts_client is not None and ts_messages is not None and self.ts_client.is_trained:
            try:
                suggestions = await self.ts_client.suggest(ts_messages, top_k=1)
                if suggestions:
                    sid = suggestions[0].id
                    if sid == INTENT_TOOL_IDS["find_one"]:
                        intent = "find_one"
                    elif sid == INTENT_TOOL_IDS["build_set"]:
                        intent = "build_set"
                    else:
                        intent = "out-of-scope"

                    result = IntentResponse(intent=intent)

                    # record (продолжаем собирать данные даже после обучения)
                    try:
                        await self.ts_client.record(
                            context=ts_messages,
                            selected_tools=[INTENT_TOOL_IDS[intent]],
                            parent_context=parent_sample_id,
                            wait=False,
                        )
                    except Exception:
                        logger.exception("ToolSuggest record failed")

                    return result
            except Exception:
                logger.exception("ToolSuggest suggest failed, falling back to LLM")

        # 2) Fallback: текущая LLM-классификация (твоя логика как была)
        prompt_parts = []
        if conversation_history:
            history_context = self._format_history_context(conversation_history)
            if history_context:
                prompt_parts.append("Previous conversation context:")
                prompt_parts.append(history_context)
                prompt_parts.append("")

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

        # 3) COLLECTION: записываем (context -> intent) в ToolSuggest
        if self.ts_client is not None and ts_messages is not None:
            try:
                await self.ts_client.record(
                    context=ts_messages,
                    selected_tools=[INTENT_TOOL_IDS[result.intent]],
                    parent_context=parent_sample_id,
                    wait=False,
                )
            except Exception:
                logger.exception("ToolSuggest record failed")

        logger.debug(f"Intent classified as: {result.intent}")
        return result
    
    async def act(self, user_input: str, conversation_history: Optional[list[dict]] = None, *, parent_sample_id=None):
        logger.info(f"[{self.name}] received input: {user_input}")

        intent_resp = await self.classify_intent(user_input, conversation_history, parent_sample_id=parent_sample_id)
        logger.info(f"Intent classified: {intent_resp.intent}")

        if intent_resp.intent == "out-of-scope":
            return {"intent": "out-of-scope", "message": "Please ask for scientific papers or studies."}

        ...
        # остальное оставь как есть (там синхронные вызовы LLM и tool_func)


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
    def _to_ts_messages(self, query: str, conversation_history: Optional[list[dict]] = None):
        from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
        msgs = []
        if conversation_history:
            for ex in conversation_history[-3:]:
                if ex.get("user"):
                    msgs.append(ModelRequest(parts=[UserPromptPart(content=ex["user"])]))
                if ex.get("assistant"):
                    msgs.append(ModelResponse(parts=[TextPart(content=ex["assistant"])]))

        msgs.append(ModelRequest(parts=[UserPromptPart(content=query)]))
        return msgs

