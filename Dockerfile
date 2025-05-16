FROM python:3.9-slim

WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install system and Python dependencies
RUN apt-get update && \
    apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]