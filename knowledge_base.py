import os
import logging
import asyncio
from typing import List, Dict, Any
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles extraction and chunking of text from PDF documents."""
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts all text from a PDF file.

        Args:
            pdf_path: The path to the PDF file.

        Returns:
            A string containing all extracted text.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Splits a long string of text into smaller, overlapping chunks.

        Args:
            text: The input text to chunk.
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The number of characters to overlap between chunks.

        Returns:
            A list of text chunks.
        """
        chunks = []
        if not text:
            return chunks

        words = text.split()
        current_chunk_words = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                
                overlap_chars = ""
                for w in reversed(current_chunk_words):
                    if len(overlap_chars) + len(w) + 1 > chunk_overlap and overlap_chars:
                        break
                    overlap_chars = w + " " + overlap_chars if overlap_chars else w
                
                current_chunk_words = overlap_chars.split()
                current_length = sum(len(w) + 1 for w in current_chunk_words) - 1 if current_chunk_words else 0
            
            current_chunk_words.append(word)
            current_length += len(word) + 1

        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        logger.info(f"Chunked text into {len(chunks)} chunks.")
        return chunks


class EmbeddingGenerator:
    """Generates embeddings using the Sentence Transformers library."""
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the Sentence Transformers model.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text.strip():
            logger.warning("Attempted to generate embedding for empty text. Skipping.")
            return []
        try:
            embedding = await asyncio.to_thread(self.embedding_model.encode, text)
            logger.debug(f"Generated embedding for text snippet.")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for text: '{text[:50]}...': {e}")
            return []


class QdrantManager:
    """Manages interaction with the Qdrant vector database."""
    def __init__(self, url: str, api_key: str, collection_name: str, chat_history_collection: str = None):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.chat_history_collection = chat_history_collection or collection_name
        self.vector_size = 384  # Embedding size for 'all-MiniLM-L6-v2'

    async def create_collection(self, collection_name: str = None):
        """Creates a Qdrant collection if it doesn't already exist, with index for user_id if needed."""
        collection_name = collection_name or self.collection_name
        try:
            collections_response = await asyncio.to_thread(self.client.get_collections)
            existing_collections = [c.name for c in collections_response.collections]

            if collection_name not in existing_collections:
                # Define collection configuration with user_id index for chat history collection
                payload_indexing_config = (
                    models.PayloadSchemaType.KEYWORD
                    if collection_name == self.chat_history_collection
                    else None
                )
                await asyncio.to_thread(
                    self.client.recreate_collection,
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                )
                if payload_indexing_config:
                    await asyncio.to_thread(
                        self.client.create_payload_index,
                        collection_name=collection_name,
                        field_name="user_id",
                        field_schema=payload_indexing_config
                    )
                    logger.info(f"Created index for 'user_id' in collection '{collection_name}'.")
                logger.info(f"Collection '{collection_name}' created.")
            else:
                # Check if user_id index exists for chat history collection
                if collection_name == self.chat_history_collection:
                    try:
                        await asyncio.to_thread(
                            self.client.create_payload_index,
                            collection_name=collection_name,
                            field_name="user_id",
                            field_schema=models.PayloadSchemaType.KEYWORD
                        )
                        logger.info(f"Ensured index for 'user_id' in collection '{collection_name}'.")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"Index for 'user_id' already exists in collection '{collection_name}'.")
                        else:
                            logger.error(f"Error ensuring index for 'user_id' in collection '{collection_name}': {e}")
                logger.info(f"Collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection '{collection_name}': {e}")

    async def upsert_vectors(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]], collection_name: str = None):
        """
        Inserts or updates vectors (embeddings) and their associated text/metadata in Qdrant.

        Args:
            texts: List of text chunks.
            embeddings: List of embedding vectors corresponding to the text chunks.
            metadata: List of metadata dictionaries for each text chunk.
            collection_name: Optional specific collection to upsert into.
        """
        collection_name = collection_name or self.collection_name
        if not texts or not embeddings or len(texts) != len(embeddings) or len(texts) != len(metadata):
            logger.warning("Mismatch in lengths of texts, embeddings, or metadata. Skipping upsert.")
            return

        points = [
            models.PointStruct(
                id=str(hash(text + str(meta.get("timestamp", "")))),  # Unique ID based on text and timestamp
                vector=emb,
                payload={"text": text, **meta}
            )
            for text, emb, meta in zip(texts, embeddings, metadata)
        ]

        try:
            operation_info = await asyncio.to_thread(
                self.client.upsert,
                collection_name=collection_name,
                wait=True,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to Qdrant collection '{collection_name}'. Status: {operation_info.status}")
        except Exception as e:
            logger.error(f"Error upserting vectors to Qdrant collection '{collection_name}': {e}")

    async def search_vectors(self, query_embedding: List[float], top_k: int = 3, collection_name: str = None, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Searches the Qdrant collection for relevant text chunks based on a query embedding.

        Args:
            query_embedding: The embedding of the user's query.
            top_k: The number of top relevant results to retrieve.
            collection_name: Optional specific collection to search.
            user_id: Optional user ID to filter results.

        Returns:
            A list of dictionaries, each containing the retrieved text and its metadata.
        """
        collection_name = collection_name or self.collection_name
        if not query_embedding:
            logger.warning(f"Query embedding is empty. Cannot perform Qdrant search in collection '{collection_name}'.")
            return []
        try:
            filter_condition = models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ) if user_id else None

            search_result = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=filter_condition,
                limit=top_k,
                with_payload=True
            )
            results = [hit.payload for hit in search_result]
            logger.info(f"Retrieved {len(results)} results from Qdrant search in collection '{collection_name}'.")
            return results
        except Exception as e:
            logger.error(f"Error searching Qdrant collection '{collection_name}': {e}")
            return []

async def build_knowledge_base_from_pdf(
    pdf_path: str,
    pdf_processor: PDFProcessor,
    embedding_generator: EmbeddingGenerator,
    qdrant_manager: QdrantManager
):
    """
    Orchestrates the process of extracting text from a PDF, generating embeddings,
    and storing them in Qdrant.
    """
    logger.info(f"Building knowledge base for PDF: {pdf_path}")
    text = pdf_processor.extract_text_from_pdf(pdf_path)
    if not text:
        logger.warning(f"No text extracted from {pdf_path}. Skipping knowledge base build for this PDF.")
        return

    chunks = pdf_processor.chunk_text(text)
    if not chunks:
        logger.warning(f"No chunks generated from {pdf_path}. Skipping knowledge base build for this PDF.")
        return

    embeddings = []
    valid_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = await embedding_generator.generate_embedding(chunk)
        if embedding:
            embeddings.append(embedding)
            valid_chunks.append(chunk)
        else:
            logger.warning(f"Could not generate embedding for chunk {i+1} of {pdf_path}. Skipping this chunk.")

    if not embeddings:
        logger.error(f"No valid embeddings generated for {pdf_path}. Cannot upsert to Qdrant.")
        return

    pdf_filename = os.path.basename(pdf_path)
    metadata = [{"source": pdf_filename, "chunk_index": i} for i in range(len(embeddings))]

    await qdrant_manager.upsert_vectors(valid_chunks, embeddings, metadata)
    logger.info(f"Knowledge base updated for {pdf_path} with {len(valid_chunks)} chunks.")