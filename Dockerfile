FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install system dependencies (keep these!)
RUN apt-get update && \
    apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]