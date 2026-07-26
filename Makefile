.PHONY: agent dry-run help infer local-mlflow-up pipeline-fact pipeline-recommendations start-arcadedb start-neo4j stop-arcadedb stop-neo4j train full-docs

-include .env

GNN_SRC := graph-neural-network/src

help:
	@echo "Target disponibili:"
	@echo "  make train                    - addestra il modello GNN con logging MLflow"
	@echo "  make dry-run                  - esegue la pipeline GNN senza MLflow"
	@echo "  make infer                    - esegue l'inferenza GNN da MLflow"
	@echo "  make agent                    - avvia l'entry point Graph RAG"
	@echo "  make pipeline-fact            - carica il grafo Neo4j dall'ETL"
	@echo "  make pipeline-recommendations - importa le raccomandazioni in Neo4j"
	@echo "  make start-neo4j              - avvia Neo4j in Docker"
	@echo "  make stop-neo4j               - arresta Neo4j in Docker"
	@echo "  make full-docs                - avvia la documentazione del GNN"
	@echo "  make local-mlflow-up          - avvia MLflow locale"

train:
	PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys

dry-run:
	PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys --dry-run

infer:
	PYTHONPATH=$(GNN_SRC) uv run python -m graph_recc_sys --no-training

agent:
	uv run python graph-rag-agent/main.py

pipeline-fact:
	uv run python pipeline/ETL_fact.py

pipeline-recommendations:
	uv run python pipeline/ETL_recc.py

start-neo4j:
	docker compose up -d neo4j

stop-neo4j:
	docker compose down

full-docs:
	$(MAKE) -C graph-neural-network/docs live

local-mlflow-up:
	uv run mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_URI) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT)
