# chatbot_agent.py
import logging
from typing import List, Dict, Any
import google.generativeai as genai
import asyncio
import uuid # For generating unique IDs for chat turns
import time # For timestamp in history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotAgent:
    """
    Handles conversational logic, RAG (Retrieval Augmented Generation),
    and interaction with the Google Gemini LLM.
    Also manages chat history using Qdrant.
    """
    def __init__(self, api_key: str, knowledge_qdrant_manager, history_qdrant_manager, embedding_generator):
        genai.configure(api_key=api_key)
        self.generative_model = genai.GenerativeModel('gemini-1.5-flash')
        self.knowledge_qdrant_manager = knowledge_qdrant_manager # For RAG context
        self.history_qdrant_manager = history_qdrant_manager # For chat history
        self.embedding_generator = embedding_generator
        self.system_prompt = """
        You are DJS CodeAI, the official chatbot for the Artificial Intelligence Club at Dwarkadas J. Sanghvi College of Engineering.
        Your purpose is to provide helpful and accurate information about the DJS CodeAI club.
        You are a friendly, enthusiastic, informative, and encouraging assistant.
        Always draw upon the retrieved context from the club's knowledge base to answer questions if relevant.
        If the retrieved context does not contain the answer, state that you don't have that specific information.
        Avoid making up facts or fabricating details.
        For information not found in the context (like specific event timings or application processes), kindly suggest checking official club announcements, the college website, or contacting club organizers directly.
        Your responses should consistently reinforce the exciting opportunities and transformative impact that joining DJS CodeAI can have.
        """

    async def _save_chat_turn(self, session_id: str, role: str, content: str, turn_index: int):
        """Saves a single chat turn to the Qdrant history collection."""
        try:
            turn_text = f"{role}: {content}"
            embedding = await self.embedding_generator.generate_embedding(turn_text)
            if embedding:
                metadata = {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "turn_index": turn_index,
                    "timestamp": time.time() # Use time.time() directly as it's called in async context
                }
                await self.history_qdrant_manager.upsert_vectors(
                    texts=[turn_text],
                    embeddings=[embedding],
                    metadata=[metadata]
                )
                logger.debug(f"Saved chat turn {turn_index} for session {session_id}.")
            else:
                logger.warning(f"Failed to generate embedding for chat turn. Not saving history for turn {turn_index}.")
        except Exception as e:
            logger.error(f"Error saving chat turn to history: {e}")

    async def _retrieve_chat_history(self, session_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
        """Retrieves recent chat history for a given session ID from Qdrant."""
        try:
            # Generate an embedding for a generic query to use for search,
            # then filter by session_id in payload.
            # This is a common pattern when you want to retrieve based on metadata.
            dummy_query_vector = await self.embedding_generator.generate_embedding("retrieve chat history")
            if not dummy_query_vector:
                logger.warning("Could not generate dummy embedding for history retrieval. Skipping.")
                return []

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            )

            # Use search_vectors from history_qdrant_manager, passing the filter
            search_results = await self.history_qdrant_manager.search_vectors(
                query_embedding=dummy_query_vector,
                top_k=max_turns * 2, # Fetch more to sort and get the latest
                user_id=session_id # Pass session_id to search_vectors for filtering
            )

            history = []
            for hit_payload in search_results:
                history.append({
                    "role": hit_payload.get("role"),
                    "content": hit_payload.get("content"),
                    "turn_index": hit_payload.get("turn_index"),
                    "timestamp": hit_payload.get("timestamp")
                })
            
            # Sort by turn_index to maintain chronological order
            history.sort(key=lambda x: x.get("turn_index", 0))
            
            logger.debug(f"Retrieved {len(history)} chat turns for session {session_id}.")
            return history[-max_turns:] # Return only the most recent turns
        except Exception as e:
            logger.error(f"Error retrieving chat history for session {session_id}: {e}")
            return []

    async def generate_response(self, user_message: str, session_id: str) -> str:
        """
        Generates a conversational response using RAG and maintains chat history.

        Args:
            user_message: The current message from the user.
            session_id: A unique identifier for the conversation session.

        Returns:
            The generated response from the LLM.
        """
        try:
            # Retrieve recent chat history
            chat_history = await self._retrieve_chat_history(session_id)
            
            # Determine the next turn index based on retrieved history
            next_turn_index = len(chat_history) # Each turn is a user-model pair, so count total messages

            # Save current user message to history
            await self._save_chat_turn(session_id, "user", user_message, next_turn_index)

            # 1. Generate embedding for the user's query
            query_embedding = await self.embedding_generator.generate_embedding(user_message)
            retrieved_context = ""
            if query_embedding:
                # 2. Retrieve relevant context from Qdrant knowledge base
                retrieved_chunks = await self.knowledge_qdrant_manager.search_vectors(query_embedding, top_k=3)
                context_texts = [chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")]
                if context_texts:
                    retrieved_context = "\n\nRelevant Information from DJS CodeAI Knowledge Base:\n" + "\n".join(context_texts)
                    logger.info("Context retrieved from Qdrant knowledge base.")
                else:
                    logger.info("No relevant context found in Qdrant knowledge base for the query.")
            else:
                logger.warning("Could not generate embedding for user query. No RAG context will be used.")

            # 3. Construct the full prompt for the LLM
            full_prompt_parts = [
                {"role": "user", "parts": [self.system_prompt]},
            ]

            # Add previous chat history for the LLM to maintain context
            for turn in chat_history:
                full_prompt_parts.append({"role": turn["role"], "parts": [turn["content"]]})

            # Add retrieved context as part of the current turn's information
            if retrieved_context:
                full_prompt_parts.append({"role": "user", "parts": ["Here is some additional information that might be relevant:\n" + retrieved_context]})

            # Finally, add the current user message
            full_prompt_parts.append({"role": "user", "parts": [user_message]})

            # 4. Generate response using Gemini
            response = await asyncio.to_thread(self.generative_model.generate_content, full_prompt_parts)
            response_text = response.text
            logger.info(f"Generated response: {response_text[:100]}...")

            # Save model's response to history
            await self._save_chat_turn(session_id, "model", response_text, next_turn_index + 1)

            return response_text
        except Exception as e:
            logger.exception("Error generating chatbot response:")
            return "I apologize, but I encountered an error while trying to generate a response. Please try again later."