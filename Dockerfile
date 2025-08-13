# Use the vLLM OpenAI image: https://hub.docker.com/r/vllm/vllm-openai
FROM vllm/vllm-openai:gptoss


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application code
COPY main.py .


# Set environment variables for Datadog tracing
# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Override the base image's entrypoint to run our FastAPI app
ENTRYPOINT []

# Run the FastAPI app via uvicorn (Cloud Run uses $PORT)
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 