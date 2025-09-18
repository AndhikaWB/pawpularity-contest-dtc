MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --always-make

.ONESHELL:

# -----

help:
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST) | column -tl 2

setup: ## Setup the Python dependencies (CUDA)
	uv sync --dev --group cuda --group monitoring

setup-cpu: ## Setup the Python dependencies (CPU)
	uv sync --dev --group cpu --group monitoring

compose-up: ## Run the Docker services that are required by most scripts
	cd infra/docker-services
	docker compose --env-file compose.env up

compose-down: ## Stop the Docker services
	cd infra/docker-services
	rm -rf home/
	rm -rf var/
	docker compose --env-file compose.env down

# -----

preprocess: ## Preprocess the raw data and commit to lakeFS
	uv run src/pawpaw/preprocess.py

training: ## Train a model based on the latest data (unused, use evaluation instead)
	uv run src/pawpaw/training.py

evaluation: ## Evaluate model performance and train a new model if necessary
	uv run src/pawpaw/evaluation.py

prep-eval: ## Preprocess data, and evaluate/train model in one go
	$(MAKE) preprocess && $(MAKE) evaluation

server: ## Run the model prediction server (without Docker)
	uv run fastapi run --port 8765 src/pawpaw/server.py

webapp: ## Run the web app server (without Docker)
	uv run streamlit run --browser.gatherUsageStats false --server.port 8501 src/pawpaw/webapp.py

pytest: ## Run Pytest on the current state of the code
	uv run pytest tests/

# -----

server-build: ## Build the Docker image for model prediction server
	docker build -f infra/docker-deployment/server.Dockerfile -t pawpaw/server:latest .

server-run: ## Run the model prediction server image
	docker run -it --rm --name pawpaw-server --network pawpaw_default -p 8765:8765 \
		--env-file .env --env-file infra/docker-deployment/override.env \
		pawpaw/server:latest

webapp-build: ## Build the Docker image for web app server
	docker build -f infra/docker-deployment/webapp.Dockerfile -t pawpaw/webapp:latest .

webapp-run: ## Run the web app server image
	docker run -it --rm --name pawpaw-webapp --network pawpaw_default -p 8501:8501 \
		--env-file .env --env-file infra/docker-deployment/override.env \
		pawpaw/webapp:latest

# -----

workflow-build: ## Build the workflow image that will be deployed to Prefect
	docker build -f infra/docker-deployment/workflow.Dockerfile -t pawpaw/workflow:latest .

workflow-run: ## To test the workflow image before actually deploying to Prefect
	docker run -it --rm --name pawpaw-workflow --network pawpaw_default \
		--env-file .env --env-file infra/docker-deployment/override.env \
		pawpaw/workflow:latest

workflow-deploy: ## Deploy the workflow image to Prefect
	uv run src/pawpaw_prefect/deploy.py