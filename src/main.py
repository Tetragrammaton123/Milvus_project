from src.embeddings.build_index import build_milvus_index
from src.agents.base import OpenAILLM
from src.agents.search_agent import SearchAgent
from src.agents.combinator_agent import CombinatorAgent
from src.agents.chat_agent import ChatAgent
from sentence_transformers import SentenceTransformer

def main():
    collection, embedding_model = build_milvus_index()

    API_KEY = "RD_os_intent_classifier:Roman_Golubev:5a4fa32142e54a4db89018c03ac3cad3"
    BASE_URL = "http://31.56.222.86:8002/api/providers/openai/v1"

    llm = OpenAILLM(api_key=API_KEY, base_url=BASE_URL, model="gpt-4o-mini")

    search_agent = SearchAgent(name="search_agent", role="retriever", llm=llm, embedding_model=embedding_model, collection=collection)
    combinator_agent = CombinatorAgent(name="combinator_agent", role="expander", llm=llm)
    chat_agent = ChatAgent(name="chat_agent", role="orchestrator", llm=llm)

    chat_agent.add_tool("classify_intent", chat_agent.classify_intent)
    chat_agent.add_tool("search", search_agent.search)
    chat_agent.add_tool("generate_queries", combinator_agent.generate_queries)

    result = chat_agent.act("Give me pizza")
    print("\nFinal structured result:")
    print(result)

if __name__ == "__main__":
    main()
