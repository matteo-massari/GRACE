# Getting started

Run commands from the repository root. Python 3.12 and `uv` are required;
Docker is needed only for Neo4j and MLflow. For Windows, use WSL2 for Make
commands or the equivalent Python commands in the root README.

## 1. Prepare the environment

```bash
git clone https://github.com/matteo-massari/GRACE.git
cd GRACE
uv sync --locked
cp .envexample .env
```

Copy `.envexample` only during initial setup. Preserve existing configuration.
Secrets, generated datasets, model artifacts, caches, and HTML are excluded from Git.

## 2. Train without external services

```bash
make smoke-train EPOCHS=10
```

This generates synthetic data and runs training, validation, and testing with
`smoke.yaml` and the `full` sampler. `--dry-run` disables MLflow logging; it still
trains the model. The demo does not save an MLflow model or export recommendations.

The default device is `auto`: CUDA first, then Apple MPS, then CPU.
Use `make smoke-train DEVICE=cpu` to force CPU execution. Other training commands
accept `--device auto`, `cpu`, `mps`, `cuda`, or `cuda:N` through `ARGS`.
Explicit unavailable devices fail before data loading. CUDA requires a compatible
PyTorch installation and NVIDIA driver; MPS requires a supported Apple GPU/build.
Use CPU if your accelerator lacks a required PyTorch/PyG operation.

## 3. Configure and start services

| Variable | Meaning / default |
| --- | --- |
| `DATA_PATH` | GNN data root: `database/gnn` |
| `NEO4J_HTTP_PORT` | Browser port: `7474` |
| `NEO4J_BOLT_PORT` | Driver port: `7687` |
| `NEO4J_URI` | `bolt://localhost:7687`; match the Bolt port |
| `NEO4J_USERNAME` | Initial username: `neo4j` |
| `NEO4J_PASSWORD` | Must match the password in an existing database volume |
| `NEO4J_DATABASE` | `neo4j` |
| `MLFLOW_PORT` | Tracking server port: `1234` |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:1234`; match `MLFLOW_PORT` |
| `MLFLOW_EXPERIMENT` | Client experiment name |
| `AZURE_OPENAI_API_KEY` | LLM credentials, needed only for the agent |
| `AZURE_OPENAI_ENDPOINT` | Base URL of the compatible LLM service |
| `RECOMMENDATIONS_CSV_PATH` | Optional override for the recommendation import path |

```bash
make start-neo4j
make start-mlflow
```

The separate Compose files start services in the background and wait for health
checks. Neo4j Browser defaults to `http://127.0.0.1:7474`; MLflow defaults to
`http://127.0.0.1:1234`. If a port is occupied, change it in `.env`. Changing
Neo4j's HTTP port does not affect Bolt; changing Bolt or MLflow ports requires
updating the corresponding client URI.

Both services persist data in Docker volumes. MLflow stores SQLite and artifacts
in its own volume and transfers artifacts over HTTP. Earlier local experiments
are not migrated automatically. Server paths and bind settings are defined in
the Compose file; legacy `MLFLOW_BACKEND_URI`, `MLFLOW_ARTIFACT_ROOT`, and
`MLFLOW_HOST` settings are not used there.

## 4. Load the knowledge graph

```bash
make synthetic-data
make pipeline-synthetic
```

The loader imports customers, products, sales representatives, hierarchy, and
purchases. It excludes placeholder recommendations by default. No private Impala
connector is required for this demo.

## 5. Train and save a model

```bash
make train ARGS="--config graph-neural-network/smoke.yaml --sampler full --device auto --epochs 10"
```

MLflow records hyperparameters, losses, metrics, epoch checkpoints, and
`final_model`. Omit `--config` to use the larger `interface.yaml` configuration.
The default `neighbor` sampler requires a compatible `pyg-lib` installation.
`full` keeps all nodes and includes only edges preceding the first batch target;
it needs no sampling extension but is intended for small datasets.

```bash
uv run python -m graph_recc_sys --help
```

Other options include `--epochs`, `--seed`, `--dry-run`, and `--no-training`.
Graph settings use `n_customers`, `n_months`, and `min_connections`. Existing YAML
files using the old Italian keys remain accepted, but cannot mix both names
for the same setting.

## 6. Generate and import recommendations

Copy the completed run ID from MLflow and edit the existing settings block in
the same YAML configuration used for training:

```yaml
settings:
  mlflow:
    run_id: "SAVED_RUN_ID"
    model_name: "final_model"
```

Keep `hyperparameters`, data, and mappings consistent with the trained model.
The client experiment name comes from `MLFLOW_EXPERIMENT`.

```bash
make infer ARGS="--config graph-neural-network/smoke.yaml --sampler full --device auto"
make pipeline-recommendations
```

The demo export is `database/gnn/rec_tables/10/dataset_res_10_features_True.csv`,
with `ids_customer`, `ids_product`, and `product_name`. Set
`RECOMMENDATIONS_CSV_PATH` if the path changes. Export uses the test context;
it does not retrain on all purchases. Imports update relationships without
removing earlier recommendations. See [pipeline behavior](project/graph_doc.md).

## 7. Ask the agent

Configure the LLM credentials and model access described in the
[agent guide](project/agent_doc.md), then run:

```bash
make agent ARGS="Which customers in Roma have the most purchases?"
```

The CLI answers one question and exits. It defaults to English and follows the
user's language. Complete step 6 before asking about GNN recommendations.
`make agent ARGS="--help"` displays usage without service connections.

## Checks, documentation, and shutdown

```bash
make check
make docs
make full-docs
```

`make docs` builds English at `docs/_build/html/index.html` and Italian at
`docs/_build/html/it/index.html`, treating Sphinx warnings as errors. The language
switch opens the matching page. Edit English Markdown in `docs/` and Italian
prose in `docs/it/`; theme configuration, API docstrings, and common images are shared.
`make full-docs` builds both languages and serves them at `http://localhost:8000`.
Rebuild after source edits; stop the server with Ctrl+C.

```bash
make stop-neo4j
make stop-mlflow
```

These commands stop services independently and retain their volumes.
`make local-mlflow-up` and `mlflow_init.sh` alias MLflow startup.
See the [verification report](../verification.md) for tested behavior.
