# Official lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables for non-interactive installs and unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

WORKDIR $HOME/app

# Install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (compact size) then project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files with ownership assigned to user
COPY --chown=user:user . $HOME/app

# Switch to non-root user
USER user

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Launch FastAPI web application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
