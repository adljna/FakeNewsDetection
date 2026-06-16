FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

ENV PORT=8080
ENV MODEL_PATH=/app/model/pipeline_v1.pkl

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 120"]