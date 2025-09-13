MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --always-make

.ONESHELL:

# -----

setup:
	uv sync --dev --group cuda --group monitoring

setup-cpu:
	uv sync --dev --group cpu --group monitoring

compose-up:
	cd infra/docker-services
	docker compose --env-file compose.env up

compose-down:
	cd infra/docker-services
	rm -rf home/
	rm -rf var/
	docker compose --env-file compose.env down

# -----

preprocess:
	uv run src/pawpaw/preprocess.py

training:
	uv run src/pawpaw/training.py

evaluation:
	uv run src/pawpaw/evaluation.py

server:
	uv run fastapi run --port 8765 src/pawpaw/server.py

webapp:
	uv run streamlit run --browser.gatherUsageStats false --server.port 8501 src/pawpaw/webapp.py

pytest:
	uv run pytest tests/

# -----

server-build:
	docker build -f infra/docker-deployment/server.Dockerfile -t pawpaw/server:latest .

server-run:
	docker run -it --rm --name pawpaw-server --network pawpaw_default -p 8765:8765 \
		--env-file .env --env-file infra/docker-deployment/override.env \
		pawpaw/server:latest

webapp-build:
	docker build -f infra/docker-deployment/webapp.Dockerfile -t pawpaw/webapp:latest .

webapp-run:
	docker run -it --rm --name pawpaw-webapp --network pawpaw_default -p 8501:8501 \
		--env-file .env --env-file infra/docker-deployment/override.env \
		pawpaw/webapp:latest

# -----

workflow-build:
	docker build -f infra/docker-deployment/workflow.Dockerfile -t pawpaw/workflow:latest .

workflow-run: # To test the container before actually deploying to Prefect
	docker run -it --rm --name pawpaw-workflow --network pawpaw_default \
		--env-file .env --env-file infra/docker-deployment/override.env \
		pawpaw/workflow:latest

workflow-deploy:
	uv run src/pawpaw_prefect/deploy.py