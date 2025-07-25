# main.py
import os
import asyncio
import threading
import time # Import time for sleep in watcher thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Keep CORS for frontend
from pydantic import BaseModel
import logging
import uuid # For generating session IDs

# Import components from your project files
from config import Config
from knowledge_base import PDFProcessor, EmbeddingGenerator, QdrantManager, build_knowledge_base_from_pdf
from chatbot_agent import ChatbotAgent
# Import the directory watcher
from directory_watcher import start_watching_pdfs

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DJS CodeAI Chatbot Backend",
    description="A conversational AI agent for DJS CodeAI club with RAG and dynamic PDF knowledge base.",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests (as it was in your file)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins in production, e.g., ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
pdf_processor: PDFProcessor = None
embedding_generator: EmbeddingGenerator = None
knowledge_qdrant_manager: QdrantManager = None # For knowledge base
history_qdrant_manager: QdrantManager = None   # For chat history
chatbot_agent: ChatbotAgent = None
dir_watcher_thread: threading.Thread = None
observer = None # Watchdog observer instance

class ChatRequest(BaseModel):
    """Pydantic model for incoming chat requests."""
    user_message: str
    session_id: str = None # Unique ID for the conversation session (optional, generated if not provided)

@app.on_event("startup")
async def startup_event():
    """
    Initializes all components and sets up the knowledge base
    and directory watcher on application startup.
    """
    global pdf_processor, embedding_generator, knowledge_qdrant_manager, history_qdrant_manager, chatbot_agent, observer, dir_watcher_thread

    logger.info("Starting up DJS CodeAI Chatbot Backend...")

    # Ensure PDF directory exists
    if not os.path.exists(Config.PDF_DIR):
        os.makedirs(Config.PDF_DIR)
        logger.info(f"Created PDF directory: {Config.PDF_DIR}")

    # Initialize components
    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator(api_key=Config.GOOGLE_API_KEY) # Pass API key here
    
    # Initialize QdrantManager for knowledge base
    knowledge_qdrant_manager = QdrantManager(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.QDRANT_COLLECTION_NAME
    )

    # Initialize QdrantManager for chat history
    history_qdrant_manager = QdrantManager(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.QDRANT_CHAT_HISTORY_COLLECTION # Use the new history collection name
    )
    
    chatbot_agent = ChatbotAgent(
        api_key=Config.GOOGLE_API_KEY,
        knowledge_qdrant_manager=knowledge_qdrant_manager,
        history_qdrant_manager=history_qdrant_manager, # Pass history manager
        embedding_generator=embedding_generator
    )

    # Create Qdrant collections
    await knowledge_qdrant_manager.create_collection()
    await history_qdrant_manager.create_collection() # Create history collection

    # Build initial knowledge base from existing PDFs
    logger.info(f"Scanning existing PDFs in '{Config.PDF_DIR}' for initial knowledge base build...")
    existing_pdfs = [f for f in os.listdir(Config.PDF_DIR) if f.lower().endswith('.pdf')]
    if existing_pdfs:
        for filename in existing_pdfs:
            pdf_path = os.path.join(Config.PDF_DIR, filename)
            # Pass knowledge_qdrant_manager to the builder
            await build_knowledge_base_from_pdf(pdf_path, pdf_processor, embedding_generator, knowledge_qdrant_manager)
        logger.info("Initial knowledge base build complete.")
    else:
        logger.info(f"No existing PDFs found in '{Config.PDF.DIR}'. Knowledge base is empty initially.")


    # Start the directory watcher in a separate thread
    def run_watcher_in_thread():
        """Function to run the watchdog observer in a separate thread."""
        global observer
        observer = start_watching_pdfs(
            Config.PDF_DIR,
            # Pass the knowledge_qdrant_manager to the builder function
            lambda pdf_path: asyncio.run(build_knowledge_base_from_pdf(
                pdf_path, pdf_processor, embedding_generator, knowledge_qdrant_manager
            ))
        )
        try:
            while True:
                time.sleep(1) # Keep the thread alive
        except KeyboardInterrupt:
            observer.stop()
        finally:
            observer.join()

    # Start the watcher in a daemon thread so it exits cleanly with the main process
    dir_watcher_thread = threading.Thread(target=run_watcher_in_thread, daemon=True)
    dir_watcher_thread.start()
    logger.info("Directory watcher started in background thread.")
    logger.info("DJS CodeAI Chatbot Backend is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Stops the directory watcher on application shutdown."""
    global observer
    logger.info("Shutting down DJS CodeAI Chatbot Backend...")
    if observer:
        observer.stop()
        observer.join()
        logger.info("Directory watcher stopped.")
    logger.info("DJS CodeAI Chatbot Backend shutdown complete.")


@app.get("/")
async def read_root():
    """Root endpoint for health check."""
    return {"message": "Welcome to DJS CodeAI Chatbot Backend! Use /chat to interact."}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles conversational interactions with the DJS CodeAI chatbot.
    """
    # Generate a session ID if not provided by the client
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    
    try:
        response_text = await chatbot_agent.generate_response(
            user_message=request.user_message,
            session_id=session_id # Pass session_id instead of chat_history
        )
        return {"response": response_text, "session_id": session_id}
    except Exception as e:
        logger.exception("Error during chat interaction:")
        raise HTTPException(status_code=500, detail="Internal server error during chat.")