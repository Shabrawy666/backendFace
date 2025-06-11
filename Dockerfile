# Stage 1: Builder
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgl1-mesa-dev \
    libglib2.0-dev \
    && pip install --user --no-cache-dir -r requirements.txt \
    && apt-get purge -y gcc python3-dev libgl1-mesa-dev libglib2.0-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app.py .
COPY routes/ routes/
COPY core/ core/
COPY ml_service.py .

# Copy the anti-spoofing src directory to root level
COPY Silent-Face-Anti-Spoofing-master/ Silent-Face-Anti-Spoofing-master/

# Copy everything else
COPY . .

ENV PATH=/root/.local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]