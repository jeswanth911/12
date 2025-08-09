# Dockerfile

# --- Build Stage ---
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

# Install build deps for python-magic and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libmagic-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install pipenv or poetry if needed, or just install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# --- Test Stage (optional) ---
FROM builder AS test

RUN pip install pytest pytest-asyncio && \
    pytest tests

# --- Runtime Stage ---
FROM python:3.11-slim-bullseye

# Create a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install runtime deps (libmagic)
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# Ownership and permissions
RUN chown -R appuser:appgroup /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--lifespan", "on"]
