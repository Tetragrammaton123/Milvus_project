import chainlit as cl
from src.embeddings.build_index import build_milvus_index
from src.agents.base import OpenAILLM
from src.agents.search_agent import SearchAgent
from src.agents.combinator_agent import CombinatorAgent
from src.agents.chat_agent import ChatAgent
from dotenv import load_dotenv
from loguru import logger
import os


@cl.on_chat_start
async def start():
    """Initialize the application when a new chat session starts."""
    logger.info("Starting Chainlit chat session")
    
    # Show loading message
    msg = cl.Message(content="🔄 Initializing AI agents and loading knowledge base...")
    await msg.send()
    
    # Build Milvus index and load embedding model
    logger.info("Building Milvus index")
    collection, embedding_model = build_milvus_index()
    logger.success("Milvus index built successfully")
    
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    BASE_URL = os.getenv("BASE_URL")
    
    # Initialize LLM
    logger.info("Initializing OpenAI LLM")
    llm = OpenAILLM(api_key=API_KEY, base_url=BASE_URL, model="gpt-4o-mini")
    
    # Initialize agents
    logger.debug("Initializing agents")
    search_agent = SearchAgent(
        name="search_agent",
        role="retriever",
        llm=llm,
        embedding_model=embedding_model,
        collection=collection
    )
    combinator_agent = CombinatorAgent(
        name="combinator_agent",
        role="expander",
        llm=llm
    )
    chat_agent = ChatAgent(
        name="chat_agent",
        role="orchestrator",
        llm=llm
    )
    
    # Register tools with chat agent
    logger.debug("Registering tools with chat agent")
    chat_agent.add_tool("classify_intent", chat_agent.classify_intent)
    chat_agent.add_tool("search", search_agent.search)
    chat_agent.add_tool("generate_queries", combinator_agent.generate_queries)
    
    # Store agents and initialize conversation history in user session
    cl.user_session.set("chat_agent", chat_agent)
    cl.user_session.set("conversation_history", [])
    
    logger.success("All agents initialized successfully")
    
    # Update the message
    msg.content = "✅ Ready! Ask me about academic papers in machine learning."
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from the user."""
    logger.info(f"Received message: {message.content}")
    
    # Get the chat agent and conversation history from user session
    chat_agent = cl.user_session.get("chat_agent")
    conversation_history = cl.user_session.get("conversation_history", [])
    
    # Show thinking message
    thinking_msg = cl.Message(content="🤔 Processing your query...")
    await thinking_msg.send()
    
    try:
        # Get result from chat agent with conversation history
        result = chat_agent.act(message.content, conversation_history=conversation_history)
        
        # Handle out-of-scope queries
        if result.get("intent") == "out-of-scope":
            response_text = "❌ " + result.get("message", "Your query is out of scope of my abilities.")
            thinking_msg.content = response_text
            await thinking_msg.update()
            
            # Update conversation history
            conversation_history.append({
                "user": message.content,
                "assistant": response_text
            })
            cl.user_session.set("conversation_history", conversation_history)
            return
        
        # Display search results
        if "results" in result and result["results"]:
            papers = result["results"]
            rewritten_query = result.get("rewritten", "")
            
            # Create text elements for each paper
            elements = []
            for idx, paper in enumerate(papers, 1):
                title = paper.get("title", "Untitled")
                abstract = paper.get("abstract", "No abstract available")
                
                # Create a text element with title and abstract
                element = cl.Text(
                    name=f"{idx}. {title}",
                    content=abstract,
                    display="inline"
                )
                elements.append(element)
            
            # Update the message with results
            result_text = f"**Found {len(papers)} papers**"
            if rewritten_query:
                result_text += f"\n\n*Search query: {rewritten_query}*"
            
            thinking_msg.content = result_text
            thinking_msg.elements = elements
            await thinking_msg.update()
            
            logger.success(f"Displayed {len(papers)} papers to user")
            
            # Update conversation history with search results summary
            paper_titles = [p.get("title", "Untitled") for p in papers[:3]]  # First 3 titles
            assistant_summary = f"Found {len(papers)} papers. Top results: " + "; ".join(paper_titles)
            conversation_history.append({
                "user": message.content,
                "assistant": assistant_summary
            })
            cl.user_session.set("conversation_history", conversation_history)
        else:
            no_results_text = "No papers found for your query. Try rephrasing your question."
            thinking_msg.content = no_results_text
            await thinking_msg.update()
            
            # Update conversation history
            conversation_history.append({
                "user": message.content,
                "assistant": no_results_text
            })
            cl.user_session.set("conversation_history", conversation_history)
            
    except Exception as e:
        logger.exception("Error processing message")
        error_text = f"❌ An error occurred: {str(e)}"
        thinking_msg.content = error_text
        await thinking_msg.update()
        
        # Update conversation history even on error
        conversation_history.append({
            "user": message.content,
            "assistant": error_text
        })
        cl.user_session.set("conversation_history", conversation_history)

