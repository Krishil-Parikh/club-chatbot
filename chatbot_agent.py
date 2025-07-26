import logging
from typing import List, Dict, Any
import google.generativeai as genai
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotAgent:
    """
    Handles conversational logic, RAG (Retrieval Augmented Generation),
    and interaction with the Google Gemini LLM.
    """
    def __init__(self, api_key: str, qdrant_manager, embedding_generator):
        genai.configure(api_key=api_key)
        self.generative_model = genai.GenerativeModel('gemini-1.5-flash')
        self.qdrant_manager = qdrant_manager
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

    async def store_conversation(self, user_id: str, user_message: str, response: str):
        """
        Stores the user message and bot response in the Qdrant conversation history collection.

        Args:
            user_id: Unique identifier for the user.
            user_message: The user's input message.
            response: The bot's response.
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            conversation_text = f"User: {user_message}\nBot: {response}"
            embedding = await self.embedding_generator.generate_embedding(conversation_text)
            if not embedding:
                logger.warning("Could not generate embedding for conversation. Skipping storage.")
                return

            metadata = {
                "user_id": user_id,
                "user_message": user_message,
                "response": response,
                "timestamp": timestamp
            }
            await self.qdrant_manager.upsert_vectors(
                texts=[conversation_text],
                embeddings=[embedding],
                metadata=[metadata],
                collection_name=self.qdrant_manager.chat_history_collection
            )
            logger.info(f"Stored conversation for user {user_id} in Qdrant.")
        except Exception as e:
            logger.error(f"Error storing conversation for user {user_id}: {e}")

    async def generate_response(self, user_id: str, user_message: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Generates a conversational response using RAG, incorporating both knowledge base and conversation history.

        Args:
            user_id: Unique identifier for the user.
            user_message: The current message from the user.
            chat_history: A list of previous messages in the conversation (role, content).

        Returns:
            The generated response from the LLM.
        """
        try:
            query_embedding = await self.embedding_generator.generate_embedding(user_message)
            if not query_embedding:
                logger.warning("Could not generate embedding for user query. No RAG context will be used.")

            # Retrieve context from knowledge base
            knowledge_context = ""
            if query_embedding:
                knowledge_chunks = await self.qdrant_manager.search_vectors(query_embedding, top_k=5)
                knowledge_texts = [chunk.get("text", "") for chunk in knowledge_chunks if chunk.get("text")]
                if knowledge_texts:
                    knowledge_context = "\n\nRelevant Information from DJS CodeAI Knowledge Base:\n" + "\n".join(knowledge_texts)
                    logger.info("Knowledge context retrieved from Qdrant.")
                else:
                    logger.info("No relevant knowledge context found in Qdrant for the query.")

            # Retrieve conversation history from Qdrant
            conversation_context = ""
            if query_embedding and user_id:
                conversation_chunks = await self.qdrant_manager.search_vectors(
                    query_embedding,
                    top_k=5,
                    collection_name=self.qdrant_manager.chat_history_collection,
                    user_id=user_id
                )
                conversation_texts = [chunk.get("text", "") for chunk in conversation_chunks if chunk.get("text")]
                if conversation_texts:
                    conversation_context = "\n\nRelevant Conversation History:\n" + "\n".join(conversation_texts)
                    logger.info(f"Conversation history retrieved for user {user_id} from Qdrant.")
                else:
                    logger.info(f"No relevant conversation history found for user {user_id} in Qdrant.")

            full_prompt_parts = [
                {"role": "user", "parts": [self.system_prompt]},
            ]

            for turn in chat_history:
                full_prompt_parts.append({"role": turn["role"], "parts": [turn["content"]]})

            if knowledge_context:
                full_prompt_parts.append({"role": "user", "parts": ["Here is some additional information that might be relevant:\n" + knowledge_context]})

            if conversation_context:
                full_prompt_parts.append({"role": "user", "parts": ["Here is some relevant conversation history:\n" + conversation_context]})

            full_prompt_parts.append({"role": "user", "parts": [user_message]})

            response = await asyncio.to_thread(self.generative_model.generate_content, full_prompt_parts)
            response_text = response.text

            # Store the conversation
            await self.store_conversation(user_id, user_message, response_text)

            logger.info(f"Generated response for user {user_id}: {response_text[:100]}...")
            return response_text
        except Exception as e:
            logger.exception(f"Error generating chatbot response for user {user_id}:")
            return "I apologize, but I encountered an error while trying to generate a response. Please try again later."