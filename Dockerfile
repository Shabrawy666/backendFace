# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    pkg-config \
    && pip install --user --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc python3-dev pkg-config \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files (only if they exist)
COPY app.py .

# Copy directories only if they exist
COPY template[s] template[s] 2>/dev/null || true
COPY stati[c] stati[c] 2>/dev/null || true

# Alternative: Copy everything and exclude what you don't need
# COPY . .

ENV PATH=/root/.local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]