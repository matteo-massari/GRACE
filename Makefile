.PHONY: agent dry-run help infer local-mlflow-up pipeline-fact pipeline-recommendations pipeline-synthetic start-neo4j stop-neo4j synthetic-data train full-docs

-include .env

GNN_SRC := graph-neural-network/src
NEO4J_COMPOSE := docker compose -f docker-compose.neo4j.yml
MLFLOW_COMPOSE := docker compose -f docker-compose.mlflow.yml
EPOCHS ?= 3
DEVICE ?= auto
ARGS ?=
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  make smoke-train              - synthetic demo; auto-select CUDA/MPS/CPU"
	@echo "  make check / docs             - tests and bilingual documentation"
	@echo "  make train / dry-run / infer  - GNN pipeline; pass CLI flags through ARGS"
	@echo "  make agent                    - Graph RAG CLI; pass a question through ARGS"
	@echo "  make synthetic-data           - generate synthetic CSV datasets"
	@echo "  make pipeline-synthetic       - load synthetic data into Neo4j"
	@echo "  make pipeline-recommendations - import GNN recommendations into Neo4j"
	@echo "  make pipeline-fact            - private Impala integration template"
	@echo "  make start-neo4j / stop-neo4j  - manage the Neo4j Docker service"
	@echo "  make start-mlflow / stop-mlflow - manage the MLflow Docker service"
	@echo "  make full-docs                - build and serve both documentation languages"

train:
	PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys $(ARGS)

dry-run:
	PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys --dry-run $(ARGS)

infer:
	PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys --no-training $(ARGS)

agent:
	PYTHONPATH=graph-rag-agent/src uv run python graph-rag-agent/main.py "$(if $(strip $(ARGS)),$(ARGS),--help)"

pipeline-fact:
	uv run python pipeline/ETL_fact.py

pipeline-recommendations:
	uv run python pipeline/ETL_recc.py

synthetic-data:
	uv run python pipeline/generate_synthetic_database.py

pipeline-synthetic:
	uv run python pipeline/load_synthetic_neo4j.py

start-neo4j:
	$(NEO4J_COMPOSE) up -d --wait

stop-neo4j:
	$(NEO4J_COMPOSE) down

full-docs: docs
	uv run python -m http.server 8000 --directory docs/_build/html

# Compatibility alias used by mlflow_init.sh.
local-mlflow-up: start-mlflow

.PHONY: start-mlflow stop-mlflow
start-mlflow:
	$(MLFLOW_COMPOSE) up -d --wait

stop-mlflow:
	$(MLFLOW_COMPOSE) down

.PHONY: smoke-train test check docs
smoke-train: synthetic-data
	DATA_PATH=database/gnn PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys --dry-run --sampler full --device $(DEVICE) --epochs $(EPOCHS) --config graph-neural-network/smoke.yaml

test:
	PYTHONPATH=$(GNN_SRC):graph-rag-agent/src uv run python -m unittest discover -s tests -v

check: test
	uv run ruff check --select E9,F63,F7,F82 graph-neural-network/src graph-rag-agent pipeline tests

docs:
	$(MAKE) -C docs clean
	uv run sphinx-build -W --keep-going -b html docs docs/_build/html
	uv run sphinx-build -W --keep-going -b html docs/it docs/_build/html/it
	uv run python tests/check_docs_links.py
