import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to load settings from environment variables.
    Sensitive information like API keys should be stored securely
    and loaded via environment variables, not hardcoded.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "djs_codeai_knowledge")
    QDRANT_CHAT_HISTORY: str = os.getenv("QDRANT_CHAT_HISTORY", "djs_codeai_conversations")
    PDF_DIR: str = os.getenv("PDF_DIR", "data")

    # Ensure essential configurations are present for production
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    # For cloud Qdrant, QDRANT_URL and QDRANT_API_KEY are essential.
    if "https://" in QDRANT_URL and not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY environment variable must be set for cloud Qdrant instances.")