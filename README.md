# GRACE

**English** | [Italiano](README.it.md)

Graph recommendations and a conversational sales assistant by Matteo Massari.
GRACE combines a PyTorch Geometric recommendation model, a Neo4j knowledge graph,
and a Graph RAG agent. MLflow tracks training runs and model artifacts.

## Quick start

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then clone the repository:

```bash
git clone https://github.com/matteo-massari/GRACE.git
cd GRACE
uv sync --locked
cp .envexample .env  # First setup only; preserve any existing configuration.
make smoke-train
```

The demo generates 100 synthetic customers, 60 products, and 30 sales representatives,
then runs three epochs, validation, and testing. It needs no Docker, database,
LLM credentials, or MLflow server. The device is selected automatically:
**CUDA → Apple MPS → CPU**. Synthetic results verify the pipeline; they do not
measure production recommendation quality.

On Windows, use WSL2 for the Make commands, or run the Python commands directly:

```bash
uv run python pipeline/generate_synthetic_database.py
uv run python -m graph_recc_sys --dry-run --sampler full --device auto --epochs 3 --config graph-neural-network/smoke.yaml
```

`--dry-run` disables MLflow logging; training still runs. The editable installation
from `uv sync` makes both Python packages available without setting `PYTHONPATH`.

## Training and devices

```bash
make smoke-train EPOCHS=10
make smoke-train DEVICE=cpu
make train ARGS="--sampler full --device auto --epochs 10"
make dry-run ARGS="--sampler full --device mps --epochs 5"
make infer ARGS="--sampler full --device cuda:0"
```

Use `--device auto` (default), `cpu`, `mps`, `cuda`, or `cuda:N`.
An explicitly requested unavailable device produces an error before data loading.
CUDA uses the `gpu` index from the YAML configuration unless you pass `cuda:N`.
CUDA needs a compatible NVIDIA driver and CUDA-enabled PyTorch installation;
MPS needs a supported Apple GPU and PyTorch build. CPU and MPS use full precision;
CUDA mixed precision uses bfloat16 when supported and float16 otherwise.

The default `neighbor` sampler requires a compatible `pyg-lib` installation.
Use `--sampler full` for the portable small-data demo: it retains all nodes while
restricting edges to those preceding each batch. It uses more memory and is not
identical to neighbor sampling. Backend operator support depends on the installed
PyTorch/PyG versions; use `--device cpu` if an accelerator lacks a required operator.

For inference, set `settings.mlflow.run_id` and `model_name` in the YAML file.
Use the same model configuration as training, including `--config
graph-neural-network/smoke.yaml` for demo models. Exports go to `DATA_PATH/rec_tables/`.
The graph keeps the first purchase of each customer–product pair to learn new-product
recommendations. Placeholder recommendations are separate and marked `placeholder_not_gnn`.

## Services and Graph RAG

```bash
make start-neo4j
make start-mlflow
make pipeline-synthetic
make agent ARGS="Which customers in Roma have the most purchases?"
# Stop services while retaining their Docker volumes:
make stop-neo4j
make stop-mlflow
```

Configure `.env` before starting services. Neo4j uses HTTP port 7474 and Bolt port
7687; MLflow uses port 1234. Override `NEO4J_HTTP_PORT`, `NEO4J_BOLT_PORT`, or
`MLFLOW_PORT` as needed, updating `NEO4J_URI` or `MLFLOW_TRACKING_URI` to match.
Neo4j's initial username is `neo4j`; the configured password must match the database.

The agent requires `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` with access
to the configured models. It defaults to English and responds in the user's language.
It runs as a CLI, queries Neo4j, and does not execute the GNN directly. Import model
recommendations with `make pipeline-recommendations` before asking about them.

`pipeline/ETL_fact.py` is an integration template: private Impala queries and the
connector are not included. Use `make pipeline-synthetic` for the demo.

## Repository map

| Path | Purpose |
| --- | --- |
| `graph-neural-network/` | Model, training, inference, and YAML configurations |
| `graph-rag-agent/` | Conversational agent and Neo4j retrieval |
| `pipeline/` | Synthetic data, Neo4j loading, and ETL templates |
| `docs/` | English documentation, with Italian sources in `docs/it/` |
| `tests/` | Dataset, ranking, temporal-split, export, and device checks |
| `database/` | Locally generated data, excluded from Git |

## Documentation and checks

```bash
make check
make docs       # Open docs/_build/html/index.html; switch language on any page.
make full-docs  # Serve both languages locally.
make help
```

Start with the [operational guide](docs/modules/getting_started.md),
[architecture](docs/modules/project.md), and [verification report](docs/verification.md).
The Sphinx site provides an English/Italian switch that preserves the current page.
Edit Markdown sources, not generated HTML. Both languages share theme configuration,
API docstrings, and common image assets; update the corresponding prose pages when
changing documentation. Historical experiments are retained separately from current guidance.

GitHub Actions runs checks, CPU smoke training, and strict documentation builds.
Secrets, generated datasets, model artifacts, caches, and generated HTML stay out of Git.

License: [GPL-3.0](LICENSE.txt).
