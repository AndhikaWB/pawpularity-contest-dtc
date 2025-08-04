.PHONY: setup preprocess train evaluate predict compose compose_down
.ONESHELL:

setup:
	conda env create -f conda.yaml
	conda activate pytorch

preprocess:
	conda activate pytorch
	python src/preprocess.py

train:
	conda activate pytorch
	python src/training.py

evaluate:
	conda activate pytorch
	python src/evaluation.py

predict:
	conda activate pytorch
	python src/prediction.py

compose:
	cd docker
	docker compose --env-file .env.dev up

compose_down:
	cd docker
	rm -rf home
	rm -rf var
	docker compose --env-file .env.dev down