.PHONY: setup preprocess training evaluation serving webapp compose compose-down
.ONESHELL:

setup:
	uv sync

preprocess:
	uv run src/pawpaw/preprocess.py

training:
	uv run src/pawpaw/training.py

evaluation:
	uv run src/pawpaw/evaluation.py

serving:
	uv run fastapi run src/pawpaw/serving.py --port 8765

webapp:
	uv run streamlit run src/pawpaw/webapp.py

compose:
	cd docker
	docker compose --env-file .env.dev up

compose-down:
	cd docker
	rm -rf home
	rm -rf var
	docker compose --env-file .env.dev down