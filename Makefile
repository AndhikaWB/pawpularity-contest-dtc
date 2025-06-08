.PHONY: setup mlflow train
.ONESHELL:

setup:
	conda env create -f conda.yaml
	conda activate pytorch

mlflow:
	conda activate pytorch
	mlflow ui -h 127.0.0.1 -p 5000

train:
	conda activate pytorch
	python src/training.py