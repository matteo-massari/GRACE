# Verification record

## Internationalization and device selection — 4 October 2026

- Locked dependencies installed successfully with `uv sync --locked`.
- Regression tests cover data integrity, ranking, temporal batches, CSV export,
  CUDA/MPS/CPU selection, unavailable-device errors, and legacy YAML compatibility.
- Strict Sphinx builds cover both English and Italian documentation.
- One synthetic training epoch completed with automatic CPU selection, including
  validation and final test metrics.
- This runtime reports CUDA and MPS unavailable. Device-selection tests simulate
  accelerator availability; they do not establish accelerator training performance.
- No live Neo4j or LLM requests were made during this change.

## Earlier local verification — 3 October 2026

The following results were recorded in the existing project verification report;
they refer to the earlier checkout and were not all repeated for this change.

Environment: macOS Apple Silicon, Python 3.12.12, PyTorch 2.11.0,
PyTorch Geometric 2.8.0.post1, MLflow 3.14.0. Original GraphSAGE architecture
preserved; for demo use small size and sampler `full` on CPU.

## Results

- Project installation with `uv sync --locked`: successful.
- `make check`: 8 tests passed and no Python error in targeted Ruff control.
- `make docs`: build Sphinx successful with warnings treated as errors.
- `make smoke-train EPOCHS=10`: training, validation and tests completed.
- Additional training of an era with local MLflow SQLite: checkpoint and model
  saved final, then recharged for inference.
- Verified export: 1,000 recommendations, ten for each of the 100
  customers, no duplicated customer–product, identifiers and consistent names with
  synthetic graphics.
- Help of the agent and loader Neo4j executable without connections; import of
  Configuring successful agent.

## Synthetic training of ten epochs

5,453 generated events, reduced to 2,064 client pairs– distinct product. The graph
maintains the first purchase of the pair; split: 1,240 training, 412 validation,
412 tests. Seed 42 for training; configuration `graph-neural-network/smoke.yaml`.

| Final Metric on Test | Value |
| --- | ---: |
| Precision@10 | 0.08367 |
| Recall@10 | 0.19346 |
| MAP@10 | 0.06387 |
| Hit ratio@10 | 0.55102 |
| NDCG@10 | 0.13315 |
| Loss training, last era | 0.4170 |
| Loss validation, last era | 0.3981 |

It is a functional verification: these numbers do not show predictive quality on
Real customers. The model has not been optimized; loss does not show a decline
stable in this short training. The complete sampler keeps contexts
strictly prior to the first event of each batch, but is not equivalent to
sampling of the original neighbors.

## Preserved local results

Heavy outputs remain out of Git:

- `artifacts/verification/training.log`: log of the ten epochs;
- `artifacts/verification/mlflow.log`: tracking verification;
- `artifacts/verification/mlflow.db`: MLflow metadata of verification;
- `artifacts/verification/inference.yaml`: Verified model configuration;
- `mlruns/`: artifacts of local models;
- `database/gnn/rec_tables/10/dataset_res_10_features_True.csv`: model output
  from an era recharged by MLflow, distinct from placeholders.

To repeat the inference of the verified local model, from the same checkout:

```bash
MLFLOW_TRACKING_URI=sqlite:///artifacts/verification/mlflow.db \
MLFLOW_EXPERIMENT=local-verification DATA_PATH=database/gnn \
uv run python -m graph_recc_sys --no-training --sampler full --device cpu \
  --config artifacts/verification/inference.yaml
```

This CSV uses the training context to evaluate the test; it is not an export
refurbished throughout the historian for a distribution in production.

## Limit of verification

- Docker wasn't active at first. Docker services were then
  verified as described below; pipeline loading and compliance on
  databases remain unchecked live.
- LLM requests have not been sent: Agent responses require Neo4j,
  endpoints, credentials and affordable models.
- `pyg-lib` is not installed: the sampler `neighbor` and CUDA were not executed.
- Impala remains a template to complete with private queries and drivers.
- The GitHub Actions workflow was added, but not performed on GitHub.

Publishable file control did not find recognizable patterns of
API keys or private keys. `.env` remains ignored. This control is not a
complete audit of the Git history.

## Separate Compose Verification

On the same day they were introduced `docker-compose.neo4j.yml` and
`docker-compose.mlflow.yml`, removing the ArcadeDB experimental compound.

Changes made with separate projects, containers, doors and temporary volumes:

- validation of both files with `docker compose config`;
- start of Neo4j 5.26.27 and MLflow 3.15.0-full with healthchecks exceeded;
- Bolt connection, Cypher query and call `apoc.meta.data()` successful;
- creation of experiment and MLflow run, recording and reading of a metric;
- HTTP loading and downloading of an artifact with verified content.

Containers and trial volumes have been removed after verification. Services
pre-existing have not been rebooted or modified. Data volume name
Original Neo4j is preserved by the new configuration.

Door 7474 was occupied by another Neo4j: to start this project too,
configure for example `NEO4J_HTTP_PORT=7475` in `.env`.
