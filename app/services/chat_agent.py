"""
RAG Chat Agent with Dynamic Tool Calling.

Implements an agentic workflow using LangChain:
1. Defines a 'Search_Documents' tool connected to the local ChromaDB vector store.
2. Employs an LLM with tool calling to decide dynamically whether to search
   the vector database or converse directly.
3. Integrates with Redis session memory for conversational context.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.memory import chat_memory
from app.services.vector_store import query_similar_documents

logger = logging.getLogger(__name__)

# System instructions guiding the agent's routing decision
SYSTEM_PROMPT = """You are an intelligent, helpful knowledge base assistant.
You have access to a tool named `Search_Documents` which queries the user's uploaded documents in a vector database.

Routing Guidelines:
1. For everyday conversational interactions (e.g. greetings, "hello", "how are you?", "thank you", or clarifications about yourself), respond directly and politely WITHOUT calling `Search_Documents`.
2. For any factual, domain-specific, or document-related questions, use `Search_Documents` to retrieve relevant excerpts.
3. Synthesize your final answer strictly based on the retrieved context whenever possible.
4. If the retrieved context does not contain the answer, acknowledge that the knowledge base does not contain that information rather than making up facts.
5. Always cite the document source (filename and chunk/page) when referencing retrieved facts.
"""


def _create_search_tool():
    """
    Constructs the Search_Documents tool compatible with LangChain agents.
    """
    try:
        from langchain.tools import tool

        @tool("Search_Documents")
        def search_documents(query: str) -> str:
            """
            Searches the knowledge base vector store for documents and excerpts relevant to the query.
            Use this tool whenever the user asks for specific information, facts, or data that may be in uploaded documents.
            """
            logger.info("Agent executing Search_Documents with query: '%s'", query)
            docs = query_similar_documents(query=query, k=4)

            if not docs:
                return "No matching documents or relevant passages found in the knowledge base."

            formatted_results = []
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown Document")
                chunk_idx = doc.metadata.get("chunk_index", "N/A")
                content = doc.page_content.strip()
                formatted_results.append(
                    f"--- Passage {idx} [Source: {source} | Chunk: {chunk_idx}] ---\n{content}"
                )

            return "\n\n".join(formatted_results)

        return search_documents
    except ImportError:
        # Fallback if langchain tools module is not installed
        def search_documents_fallback(query: str) -> str:
            docs = query_similar_documents(query=query, k=4)
            return "\n\n".join([d.page_content for d in docs]) if docs else "No documents found."

        return search_documents_fallback


async def execute_agent_chat(session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Executes the conversational agent loop:
    1. Loads past chat history from Redis.
    2. Runs the agent with tool-calling capabilities.
    3. Captures tool invocations (e.g., whether Search_Documents was called).
    4. Persists the new exchange back to Redis.

    Args:
        session_id: Client session identifier for conversation continuity.
        user_message: User question or conversational input.

    Returns:
        Dict containing: 'answer', 'tool_calls', 'sources', 'session_id'.
    """
    # 1. Fetch conversation history from Redis
    chat_history = await chat_memory.get_messages(session_id)

    # 2. Check for OpenAI API Key
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your-openai"):
        logger.warning("OPENAI_API_KEY is not configured in .env.")
        simulated_docs = query_similar_documents(query=user_message, k=2)
        mock_sources = list({d.metadata.get("source", "knowledge_base.pdf") for d in simulated_docs})
        
        fallback_answer = (
            "*(Dev Notice: `OPENAI_API_KEY` is not set or using placeholder in `.env`)*\n\n"
            f"I received your question: **\"{user_message}\"**.\n\n"
            f"ChromaDB was queried and found {len(simulated_docs)} relevant context passage(s) "
            f"from sources: {mock_sources or ['None']}.\n\n"
            "To enable live OpenAI completions and tool calling, set a valid `OPENAI_API_KEY` in `.env`."
        )

        # Persist exchange to memory
        await chat_memory.add_user_message(session_id, user_message)
        await chat_memory.add_ai_message(session_id, fallback_answer)

        return {
            "session_id": session_id,
            "query": user_message,
            "answer": fallback_answer,
            "tool_calls": ["Search_Documents"] if simulated_docs else [],
            "sources": mock_sources,
        }

    # 3. Instantiate LangChain Agent components
    try:
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_openai import ChatOpenAI

        search_tool = _create_search_tool()
        tools = [search_tool]

        llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=settings.DEBUG,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        # 4. Invoke agent
        result = await executor.ainvoke(
            {
                "input": user_message,
                "chat_history": chat_history,
            }
        )

        answer = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Extract executed tool names and sources
        tool_calls = []
        sources = set()

        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step[0], step[1]
                tool_name = getattr(action, "tool", str(action))
                tool_calls.append(tool_name)

                # Parse any source filenames mentioned in observation
                if isinstance(observation, str):
                    for line in observation.splitlines():
                        if "Source:" in line:
                            parts = line.split("Source:")
                            if len(parts) > 1:
                                source_part = parts[1].split("|")[0].strip()
                                sources.add(source_part)

        # 5. Save exchange to Redis session memory
        await chat_memory.add_user_message(session_id, user_message)
        await chat_memory.add_ai_message(session_id, answer)

        return {
            "session_id": session_id,
            "query": user_message,
            "answer": answer,
            "tool_calls": list(dict.fromkeys(tool_calls)),  # Unique preserving order
            "sources": sorted(list(sources)),
        }

    except Exception as exc:
        logger.exception("Error executing agent chat: %s", exc)
        error_msg = f"An error occurred while generating a response: {exc}"
        return {
            "session_id": session_id,
            "query": user_message,
            "answer": error_msg,
            "tool_calls": [],
            "sources": [],
        }

