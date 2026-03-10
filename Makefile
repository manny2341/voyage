.PHONY: help install generate prepare train promote serve ui test clean

help:
	@echo ""
	@echo "Voyage — Ticket Price Forecasting"
	@echo "=================================="
	@echo "  make install    Install all dependencies"
	@echo "  make generate   Generate synthetic dataset"
	@echo "  make prepare    Run data preprocessing pipeline"
	@echo "  make train      Train all models + log to MLflow"
	@echo "  make promote    Promote best model to Production"
	@echo "  make pipeline   Full pipeline: generate → prepare → train → promote"
	@echo "  make mlflow     Start MLflow tracking server (port 5000)"
	@echo "  make serve      Start FastAPI server (port 8000)"
	@echo "  make ui         Start Streamlit UI (port 8501)"
	@echo "  make docker     Build and start all Docker containers"
	@echo "  make clean      Remove generated data and cached files"
	@echo ""

install:
	pip3 install -r requirements/training.txt
	pip3 install -r requirements/api.txt
	pip3 install -r requirements/ui.txt

generate:
	python3 data/raw/generate.py

prepare:
	python3 pipeline/prepare_data.py

train:
	python3 pipeline/train.py

promote:
	python3 pipeline/promote.py

pipeline: generate prepare train promote
	@echo "Full pipeline complete."

mlflow:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow_artifacts \
		--host 0.0.0.0 \
		--port 5050

serve:
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py --server.port 8501

docker:
	docker compose up --build

clean:
	rm -rf data/processed/*.csv data/processed/*.json
	rm -rf mlflow.db mlflow_artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
