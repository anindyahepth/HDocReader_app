# Use a Python base image with specific version for reproducibility
FROM python:3.11-slim


# Set the working directory in the container
WORKDIR /app

# Install system dependencies first (this layer is cached)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY dashboard_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r dashboard_requirements.txt

# Copy the application code
COPY . .

# Create necessary directories for persistence
RUN mkdir -p /app/images /app/lines /app/mlruns

# Set environment variables with defaults
ENV FLASK_APP_PORT=5000
ENV MLFLOW_UI_PORT=5001
ENV DASHBOARD_PORT=5002
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose all three ports
EXPOSE 5000 5001 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/')" || exit 1

# Run the Flask application
CMD ["python", "main.py"]
