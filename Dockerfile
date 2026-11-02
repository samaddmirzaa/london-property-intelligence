FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/supplementary/postcodes_london.parquet ./data/supplementary/
COPY data/processed/area/ ./data/processed/area/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

ENV PORT=8000
ENV ARROW_DEFAULT_MEMORY_POOL=system

CMD uvicorn london_property.api:app --host 0.0.0.0 --port $PORT