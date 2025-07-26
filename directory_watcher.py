import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFEventHandler(FileSystemEventHandler):
    """
    Handles file system events (creation, modification, deletion) for PDFs.
    Triggers knowledge base updates asynchronously.
    """
    def __init__(self, pdf_dir: str, knowledge_base_builder_func):
        super().__init__()
        self.pdf_dir = pdf_dir
        self.knowledge_base_builder_func = knowledge_base_builder_func
        self.processed_files = set()
        self.lock = asyncio.Lock()

    def _is_pdf(self, event_path: str) -> bool:
        """Checks if the file is a PDF."""
        return event_path.lower().endswith('.pdf')

    async def _process_pdf_async(self, pdf_path: str):
        """Asynchronously processes a PDF for knowledge base update."""
        async with self.lock:
            if pdf_path in self.processed_files:
                logger.debug(f"File {pdf_path} is already being processed or recently processed. Skipping duplicate event.")
                return
            
            self.processed_files.add(pdf_path)

            try:
                logger.info(f"Detected new or modified PDF: {pdf_path}. Processing...")
                await self.knowledge_base_builder_func(pdf_path)
                logger.info(f"Finished processing PDF: {pdf_path}.")
            except Exception as e:
                logger.error(f"Error during async PDF processing for {pdf_path}: {e}")
            finally:
                await asyncio.sleep(5)
                if pdf_path in self.processed_files:
                    self.processed_files.remove(pdf_path)

    def on_created(self, event):
        """Called when a file or directory is created."""
        if not event.is_directory and self._is_pdf(event.src_path):
            asyncio.run(self._process_pdf_async(event.src_path))

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if not event.is_directory and self._is_pdf(event.src_path):
            asyncio.run(self._process_pdf_async(event.src_path))

    def on_deleted(self, event):
        """Called when a file or directory is deleted."""
        if not event.is_directory and self._is_pdf(event.src_path):
            logger.info(f"Detected deleted PDF: {event.src_path}. Removal from knowledge base not yet fully implemented.")
            if event.src_path in self.processed_files:
                self.processed_files.remove(event.src_path)

def start_watching_pdfs(directory: str, knowledge_base_builder_func):
    """
    Starts an observer to watch for changes in the specified directory.
    This function should be run in a separate thread or process.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created PDF directory: {directory}")

    event_handler = PDFEventHandler(directory, knowledge_base_builder_func)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    logger.info(f"Started watching directory '{directory}' for PDF changes.")
    return observer