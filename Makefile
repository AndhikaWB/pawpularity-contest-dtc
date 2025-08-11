.PHONY: setup preprocess training evaluation serving webapp compose compose_down
.ONESHELL:

setup:
	conda env create -f conda.yaml
	conda activate pytorch

preprocess:
	conda activate pytorch
	python src/preprocess.py

training:
	conda activate pytorch
	python src/training.py

evaluation:
	conda activate pytorch
	python src/evaluation.py

serving:
	conda activate pytorch
	python src/serving.py

webapp:
	conda activate pytorch
	streamlit run src/webapp.py

compose:
	cd docker
	docker compose --env-file .env.dev up

compose_down:
	cd docker
	rm -rf home
	rm -rf var
	docker compose --env-file .env.dev down