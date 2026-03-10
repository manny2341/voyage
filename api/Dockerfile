# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements/api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r api.txt

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/ ./api/
COPY data/processed/ ./data/processed/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
