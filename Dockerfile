# Base Image
FROM python:3.11-slim

# Install Dependencies for Ollama
RUN apt-get update && apt-get install -y curl gnupg lsb-release git procps

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

ENV OLLAMA_MODELS=/models
RUN mkdir -p $OLLAMA_MODELS

RUN ollama serve & \
    for i in {1..10}; do \
      curl -s http://localhost:11434/api/version && break; \
      echo "Waiting for Ollama..."; \
      sleep 2; \
    done && \
    ollama pull mistral && \
    ollama pull all-minilm && \
    pkill ollama

# Install Python Dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Ports: 
# 11434 -> Ollama API
# 8000 -> FastAPI
EXPOSE 8000 11434

# Start both Ollama and FastAPI
CMD ollama serve & uvicorn app:app --host 0.0.0.0 --port 8000