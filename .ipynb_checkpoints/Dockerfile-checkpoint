# Base image: slim CPU-only Python
FROM python:3.11-slim

# Set up environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV WANDB_MODE=online

# Install system packages required for PyTorch, transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only requirements first (for Docker caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project files
COPY . /app

# Create default output directory for checkpoints
RUN mkdir -p /app/models

# Default entrypoint
ENTRYPOINT ["python", "main.py"]
CMD ["--checkpoint_dir", "models", "--lr", "2e-5", "--weight_decay", "0.01", "--warmup_ratio", "0.06"]

