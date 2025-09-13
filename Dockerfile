# Base image
FROM python:3.11-slim

# Install dependencies for Ollama
RUN apt-get update && apt-get install -y curl gnupg lsb-release

# Install Ollama
RUN curl -fsSL https://ollama.com/download.sh | sh

# Set Ollama model storage
ENV OLLAMA_MODELS=/models
RUN mkdir -p $OLLAMA_MODELS

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose ports: 
# 11434 -> Ollama API
# 8000 -> FastAPI
EXPOSE 8000 11434

# Start both Ollama and FastAPI
CMD ollama serve & uvicorn app:app --host 0.0.0.0 --port 8000