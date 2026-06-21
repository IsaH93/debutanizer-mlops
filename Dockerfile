FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir xgboost scikit-learn pandas numpy matplotlib \
    evidently mlflow fastapi "uvicorn[standard]" pydantic sqlalchemy plotly \
    scipy pyarrow

COPY . .

ENV MLFLOW_ALLOW_FILE_STORE=true
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
