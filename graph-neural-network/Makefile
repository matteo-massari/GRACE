.PHONY: train dry-run infer help full-docs local-mlflow-up

-include .env


help:
	@echo "Target disponibili:"
	@echo "  make train             - esegue il training completo con logging MLflow"
	@echo "  make dry-run           - esegue la pipeline senza logging MLflow"
	@echo "  make infer             - salta il training, carica un modello da MLflow ed esegue inferenza"
	@echo "  make full-docs         - Avviare la documentazione più dettagliata in sphinx "
	@echo "  make local-mlflow-up   - Avvia mlflow"

train:
	uv run python3 -m graph_recc_sys

dry-run:
	uv run python3 -m graph_recc_sys --dry-run

infer:
	uv run python3 -m graph_recc_sys --no-training

full-docs:
	$(MAKE) -C docs live

local-mlflow-up:
	echo "start mlflow server"
	exec mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_URI) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT)

