FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV MODEL_PATH=/app/model/pipeline_v1.pkl

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]