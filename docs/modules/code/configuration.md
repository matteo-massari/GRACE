# Configuration and tracking

Hyperparameters are in `graph-neural-network/interface.yaml`; reduced demo
uses `graph-neural-network/smoke.yaml`. `--config` select the file from CLI.
The block `hyperparameters` configure the model; `settings.mlflow` identifies
run and model to load during inference.

The client reads `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` and any credentials
remote from `.env`. The local server is managed by `docker-compose.mlflow.yml`:
backend SQLite and artifact are in Docker volume and do not require routes
locals shared with the training process.

```{automodule} graph_recc_sys.utils.hyperparams
:members:
:show-inheritance:
```

```{automodule} graph_recc_sys.utils.mlflow_config
:members:
:show-inheritance:
```
