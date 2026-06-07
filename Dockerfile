# ── Base image: official Python 3.11 slim (smaller than full python) ────────
FROM python:3.11-slim

# ── Set working directory inside container ───────────────────────────────────
WORKDIR /app

# ── Install system-level dependencies first ──────────────────────────────────
# These are needed by some Python packages (e.g. sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements BEFORE copying code (Docker layer caching) ─────────────
# If requirements don't change, Docker skips the pip install step on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy your application code ───────────────────────────────────────────────
COPY . .

# ── Create directories needed at runtime ─────────────────────────────────────
RUN mkdir -p /app/data /app/chroma_db

# ── Expose port (Streamlit will listen here) ─────────────────────────────────
EXPOSE 8000

# ── Health check (important for ECS/load balancers) ──────────────────────────
# Streamlit exposes a built-in health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/_stcore/health || exit 1

# ── Start Streamlit app ───────────────────────────────────────────────────────
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8000"]