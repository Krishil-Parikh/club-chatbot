# main.py
import os
import asyncio
# import threading # REMOVE THIS
# import time # REMOVE THIS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uuid

from config import Config
from knowledge_base import PDFProcessor, EmbeddingGenerator, QdrantManager, build_knowledge_base_from_pdf
from chatbot_agent import ChatbotAgent
# from directory_watcher import start_watching_pdfs # REMOVE THIS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DJS CodeAI Chatbot Backend",
    description="A conversational AI agent for DJS CodeAI club with RAG and dynamic PDF knowledge base.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_processor: PDFProcessor = None
embedding_generator: EmbeddingGenerator = None
knowledge_qdrant_manager: QdrantManager = None
history_qdrant_manager: QdrantManager = None
chatbot_agent: ChatbotAgent = None
# dir_watcher_thread: threading.Thread = None # REMOVE THIS
# observer = None # REMOVE THIS

@app.on_event("startup")
async def startup_event():
    global pdf_processor, embedding_generator, knowledge_qdrant_manager, history_qdrant_manager, chatbot_agent # REMOVE observer, dir_watcher_thread

    logger.info("Starting up DJS CodeAI Chatbot Backend...")

    # Ensure PDF directory exists (in /tmp/data on Vercel)
    if not os.path.exists(Config.PDF_DIR):
        os.makedirs(Config.PDF_DIR)
        logger.info(f"Created PDF directory: {Config.PDF_DIR}")

    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator(api_key=Config.GOOGLE_API_KEY)

    knowledge_qdrant_manager = QdrantManager(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.QDRANT_COLLECTION_NAME
    )

    history_qdrant_manager = QdrantManager(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.QDRANT_CHAT_HISTORY_COLLECTION
    )

    chatbot_agent = ChatbotAgent(
        api_key=Config.GOOGLE_API_KEY,
        knowledge_qdrant_manager=knowledge_qdrant_manager,
        history_qdrant_manager=history_qdrant_manager,
        embedding_generator=embedding_generator
    )

    await knowledge_qdrant_manager.create_collection()
    await history_qdrant_manager.create_collection()

    logger.info(f"Scanning existing PDFs in '{Config.PDF_DIR}' for initial knowledge base build...")
    # IMPORTANT: For Vercel serverless, PDFs must be copied into the 'data' folder
    # in your project root at deployment time. Vercel will then make them available
    # in the deployed function's bundle.
    existing_pdfs = [f for f in os.listdir(Config.PDF_DIR) if f.lower().endswith('.pdf')]
    if existing_pdfs:
        for filename in existing_pdfs:
            pdf_path = os.path.join(Config.PDF_DIR, filename)
            await build_knowledge_base_from_pdf(pdf_path, pdf_processor, embedding_generator, knowledge_qdrant_manager)
        logger.info("Initial knowledge base build complete.")
    else:
        logger.info(f"No existing PDFs found in '{Config.PDF_DIR}'. Knowledge base is empty initially.")

    # REMOVE WATCHER STARTUP LOGIC
    # logger.info("Directory watcher started in background thread.")
    logger.info("DJS CodeAI Chatbot Backend is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    # global observer # REMOVE THIS
    logger.info("Shutting down DJS CodeAI Chatbot Backend...")
    # if observer: # REMOVE THIS
    #     observer.stop() # REMOVE THIS
    #     observer.join() # REMOVE THIS
    #     logger.info("Directory watcher stopped.") # REMOVE THIS
    logger.info("DJS CodeAI Chatbot Backend shutdown complete.")


@app.get("/")
async def read_root():
    return {"message": "Welcome to DJS CodeAI Chatbot Backend! Use /chat to interact."}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    try:
        response_text = await chatbot_agent.generate_response(
            user_message=request.user_message,
            session_id=session_id
        )
        return {"response": response_text, "session_id": session_id}
    except Exception as e:
        logger.exception("Error during chat interaction:")
        raise HTTPException(status_code=500, detail="Internal server error during chat.")
