FROM python:3.9-slim

WORKDIR /app

# Dependências de sistema mínimas (se necessário para libs científicas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    COLLECT_INTERVAL_SECONDS=60 \
    DATA_PATH=/app/data/realtime/gas_price_history.csv \
    COLLECT_MODE=loop

# GOLDRUSH_API_KEY e opcionais (GOLDRUSH_CHAIN_NAME, etc.) via .env ou -e

CMD ["python", "-m", "src.data.collect_realtime_gas"]

