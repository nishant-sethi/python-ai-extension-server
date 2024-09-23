# First stage: build environment
FROM python:3.12-slim as builder

# Set the working directory inside the container
WORKDIR /app

# Install necessary system packages and tools
RUN apt-get update && apt-get install -y git python3-venv curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -o ollama-installer.sh https://ollama.com/download.sh && \
    chmod +x ollama-installer.sh && \
    ./ollama-installer.sh && \
    rm ollama-installer.sh

# Pull models
RUN ollama pull llama3.1 && ollama pull nomic-embed-text

# Clone the project from a random Git URL (replace with actual URL later)
RUN git clone https://github.com/nishant-sethi/python-ai-extension-server.git /app/project

# Set the working directory to the project directory
WORKDIR /app/project

# Copy requirements.txt first to leverage Docker layer caching for dependencies
COPY requirements.txt /app/project/requirements.txt

# Create virtual environment and install dependencies
RUN python3 -m venv .env && \
    .env/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage: production environment
FROM python:3.12-slim

# Set the working directory
WORKDIR /app/project

# Copy the virtual environment and app from the builder stage
COPY --from=builder /app/project /app/project

# Expose port 5000
EXPOSE 5000

# Use ENTRYPOINT to ensure the app always runs the same command
ENTRYPOINT ["/bin/bash", "-c", "source .env/bin/activate && python main.py"]
