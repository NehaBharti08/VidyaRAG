# Convenience targets. Everything here is a thin wrapper over `uv run` so that
# CI, the container, and a developer's shell all execute identical commands.

.DEFAULT_GOAL := help
UV := uv

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Sync dependencies and install pre-commit hooks
	$(UV) sync
	$(UV) run pre-commit install

.PHONY: health
health: ## Verify configuration and vector store connectivity
	$(UV) run vidyarag health

.PHONY: download
download: ## Fetch the OpenStax PDFs (resumable, ~415MB)
	$(UV) run vidyarag download

.PHONY: ingest
ingest: ## Parse, chunk, embed and index the corpus (offline, no API key)
	$(UV) run vidyarag ingest

.PHONY: serve
serve: ## Run the HTTP API on :8000
	$(UV) run uvicorn vidyarag.api.main:app --reload

.PHONY: test
test: ## Run the test suite (excludes tests that call paid APIs)
	$(UV) run pytest -m "not costly"

.PHONY: test-cov
test-cov: ## Run tests with a coverage report
	$(UV) run pytest -m "not costly" --cov --cov-report=term-missing

.PHONY: lint
lint: ## Check formatting and lint rules without modifying files
	$(UV) run ruff check .
	$(UV) run black --check .

.PHONY: format
format: ## Apply formatting and autofixable lint rules
	$(UV) run ruff check --fix .
	$(UV) run black .

.PHONY: typecheck
typecheck: ## Type-check the core package
	$(UV) run mypy

.PHONY: check
check: lint typecheck test ## Everything CI runs

.PHONY: qdrant-up
qdrant-up: ## Start local Qdrant (requires Docker)
	docker compose up -d

.PHONY: qdrant-down
qdrant-down: ## Stop local Qdrant
	docker compose down

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
