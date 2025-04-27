FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get remove -y build-essential cmake && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]