# Stage 1: Builder (install build tools)
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc python3-dev && \
    pip install --user -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y

# Stage 2: Runtime (slimmed down)
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
COPY . .

ENV PATH=/root/.local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=3

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]