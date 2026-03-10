FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.11.1

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "*"]
