FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies required for compiling certain python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to optimize Docker build layer caching
COPY api/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code directories
COPY api /app/api
COPY src /app/src

# Setup PYTHONPATH and port environment variables
ENV PYTHONPATH=/app
ENV API_PORT=8000
ENV API_HOST=0.0.0.0

# Expose FastAPI default port
EXPOSE 8000

# Command to start the uvicorn web server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
