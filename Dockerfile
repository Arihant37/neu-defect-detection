FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxfixes3 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
WORKDIR /app
COPY . .

# Create necessary directories
RUN mkdir -p models data/train/images data/train/labels data/valid/images data/valid/labels

# Expose ports
EXPOSE 8000 7860

# Default command: run FastAPI server
CMD ["python3", "src/deploy.py", "--host", "0.0.0.0", "--port", "8000"]