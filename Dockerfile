FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY train_model.py .
COPY Online_Retail.csv .
COPY app.py .
COPY artifacts/ ./artifacts/

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y gunicorn && pip install --no-cache-dir gunicorn==20.1.0
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/artifacts/model_artifacts.joblib

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

docker build -t PROJET_ML .
docker run -p 5000:5000 PROJET_ML






