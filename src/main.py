def main():
    from src.embeddings.build_index import build_milvus_index
    from src.agents.base import OpenAILLM
    from src.agents.search_agent import SearchAgent
    from src.agents.combinator_agent import CombinatorAgent
    from src.agents.chat_agent import ChatAgent
    from dotenv import load_dotenv
    from loguru import logger
    import os

    logger.info("Starting application")
    
    collection, embedding_model = build_milvus_index()
    
    logger.debug("Loading environment variables")
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    logger.info("Initializing OpenAI LLM with model gpt-4o-mini")
    llm = OpenAILLM(api_key=API_KEY, base_url=BASE_URL, model="gpt-4o-mini")

    logger.debug("Initializing agents")
    search_agent = SearchAgent(name="search_agent", role="retriever", llm=llm, embedding_model=embedding_model, collection=collection)
    combinator_agent = CombinatorAgent(name="combinator_agent", role="expander", llm=llm)
    chat_agent = ChatAgent(name="chat_agent", role="orchestrator", llm=llm)
    logger.success("All agents initialized")

    logger.debug("Registering tools with chat agent")
    chat_agent.add_tool("classify_intent", chat_agent.classify_intent)
    chat_agent.add_tool("search", search_agent.search)
    chat_agent.add_tool("generate_queries", combinator_agent.generate_queries)

    result = chat_agent.act("Give me pizza")
    print("\nFinal structured result:")
    print(result)

if __name__ == "__main__":
    from src.logger import setup_logging

    setup_logging()
    main()
