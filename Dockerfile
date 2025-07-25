# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (if still needed by SentenceTransformers or other libs)
# Note: If you fully switch to Google Generative AI embeddings, NLTK might not be strictly needed for tokenization.
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code and the entire data directory
COPY . .
# This line will copy the entire 'data' folder and its contents into the Docker image
COPY data/ ./data/ 

# Expose port (Vercel assigns dynamically, but specify for clarity)
EXPOSE 8000

# Command to run the FastAPI app
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
