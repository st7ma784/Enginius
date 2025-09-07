FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY main_training.py .
COPY CLAUDE.md .

# Create necessary directories
RUN mkdir -p data models outputs visualizations

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Default command runs the dashboard
CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]