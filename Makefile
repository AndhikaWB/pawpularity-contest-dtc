.PHONY: setup mlflow train
.ONESHELL:

setup:
	conda env create -f conda.yaml
	conda activate pytorch

mlflow:
	conda activate pytorch
	mlflow ui -h 127.0.0.1 -p 5000 \
		--backend-store-uri "sqlite:///mlflow/mlruns.db" \
		--default-artifact-root "mlflow-artifacts:/" \
		--artifacts-destination "file:///${CURDIR}/mlflow/mlartifacts"
		--serve-artifacts

train:
	conda activate pytorch
	python src/training.py

airflow:
	cd docker
	docker compose up

airflowd:
	cd docker
	docker compose down