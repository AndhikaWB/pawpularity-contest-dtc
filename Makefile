.PHONY: setup preprocess training evaluation serving webapp compose compose-down
.ONESHELL:

setup:
	uv sync

preprocess:
	uv run src/preprocess.py

training:
	uv run src/training.py

evaluation:
	uv run src/evaluation.py

serving:
	uv run src/serving.py

webapp:
	source "$(CURDIR)/.venv/Scripts/activate"
	streamlit run src/pawpaw/webapp.py

compose:
	cd docker
	docker compose --env-file .env.dev up

compose-down:
	cd docker
	rm -rf home
	rm -rf var
	docker compose --env-file .env.dev down