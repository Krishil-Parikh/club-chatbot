import os
import asyncio
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from config import Config
from knowledge_base import PDFProcessor, EmbeddingGenerator, QdrantManager, build_knowledge_base_from_pdf
from chatbot_agent import ChatbotAgent
from directory_watcher import start_watching_pdfs

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global instances
pdf_processor: PDFProcessor = None
embedding_generator: EmbeddingGenerator = None
qdrant_manager: QdrantManager = None
chatbot_agent: ChatbotAgent = None
dir_watcher_thread: threading.Thread = None
observer = None

# Define lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    """
    global pdf_processor, embedding_generator, qdrant_manager, chatbot_agent, observer, dir_watcher_thread

    # Startup logic
    logger.info("Starting up DJS CodeAI Chatbot Backend...")

    if not os.path.exists(Config.PDF_DIR):
        os.makedirs(Config.PDF_DIR)
        logger.info(f"Created PDF directory: {Config.PDF_DIR}")

    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator()
    
    qdrant_manager = QdrantManager(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.QDRANT_COLLECTION_NAME,
        chat_history_collection=Config.QDRANT_CHAT_HISTORY
    )
    
    chatbot_agent = ChatbotAgent(
        api_key=Config.GOOGLE_API_KEY,
        qdrant_manager=qdrant_manager,
        embedding_generator=embedding_generator
    )

    await qdrant_manager.create_collection()  # Create knowledge base collection
    await qdrant_manager.create_collection(Config.QDRANT_CHAT_HISTORY)  # Create conversation history collection

    logger.info(f"Scanning existing PDFs in '{Config.PDF_DIR}' for initial knowledge base build...")
    existing_pdfs = [f for f in os.listdir(Config.PDF_DIR) if f.lower().endswith('.pdf')]
    if existing_pdfs:
        for filename in existing_pdfs:
            pdf_path = os.path.join(Config.PDF_DIR, filename)
            await build_knowledge_base_from_pdf(pdf_path, pdf_processor, embedding_generator, qdrant_manager)
        logger.info("Initial knowledge base build complete.")
    else:
        logger.info(f"No existing PDFs found in '{Config.PDF_DIR}'. Knowledge base is empty initially.")

    def run_watcher_in_thread():
        """Function to run the watchdog observer in a separate thread."""
        global observer
        observer = start_watching_pdfs(
            Config.PDF_DIR,
            lambda pdf_path: asyncio.run(build_knowledge_base_from_pdf(
                pdf_path, pdf_processor, embedding_generator, qdrant_manager
            ))
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        finally:
            observer.join()

    dir_watcher_thread = threading.Thread(target=run_watcher_in_thread, daemon=True)
    dir_watcher_thread.start()
    logger.info("Directory watcher started in background thread.")
    logger.info("DJS CodeAI Chatbot Backend is ready!")

    try:
        yield  # Application runs here
    finally:
        # Shutdown logic
        logger.info("Shutting down DJS CodeAI Chatbot Backend...")
        if observer:
            observer.stop()
            observer.join()
            logger.info("Directory watcher stopped.")
        logger.info("DJS CodeAI Chatbot Backend shutdown complete.")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DJS CodeAI Chatbot Backend",
    description="A conversational AI agent for DJS CodeAI club with RAG and dynamic PDF knowledge base.",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    """Pydantic model for incoming chat requests."""
    user_id: str
    user_message: str
    chat_history: list[dict[str, str]] = []

@app.get("/")
async def read_root():
    """Root endpoint for health check."""
    return {"message": "Welcome to DJS CodeAI Chatbot Backend! Use /chat to interact."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles conversational interactions with the DJS CodeAI chatbot.
    """
    try:
        response_text = await chatbot_agent.generate_response(
            user_id=request.user_id,
            user_message=request.user_message,
            chat_history=request.chat_history
        )
        return {"response": response_text}
    except Exception as e:
        logger.exception(f"Error during chat interaction for user {request.user_id}:")
        raise HTTPException(status_code=500, detail="Internal server error during chat.")