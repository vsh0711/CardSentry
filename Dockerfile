FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Shell form so $PORT (injected by Render / other PaaS) expands; falls back to 8000 locally.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
