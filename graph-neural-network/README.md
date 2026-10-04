# Neural Network of Recommendation
[Italiano](README.it.md)

GNN learns customer relationships – produced with GraphSAGE and produces
top-k recommendations. It reads local CSV files and builds a PyTorch Geometric graph:
does not read Knowledge Graph Neo4j directly.

## Local Demo

From the root of the repository, after `uv sync --locked`:

```bash
make smoke-train EPOCHS=10
```

The command regenerates synthetic data and complete training, validation and testing on
the automatically selected CUDA/MPS/CPU device, without MLflow or other services. `--dry-run` means training without tracking,
Not simulation. The demo uses `smoke.yaml` (sizes 32/16/8); `interface.yaml`
preserves the original configuration (512/256/128, 50 epochs).

## Training with MLflow

Configure `.env` from example to root, with `DATA_PATH=database/gnn` and
`MLFLOW_TRACKING_URI` consistent with the service door:

```bash
make synthetic-data
make start-mlflow
make train ARGS="--config graph-neural-network/smoke.yaml --sampler full --device cpu --epochs 10"
```

MLflow saves checkpoint, final model, hyperparameters and metrics. The server
Docker manages databases and artifacts in its volume; the client uses HTTP.
For the original size, omit `--config`.

## Inference and import

In the YAML configuration used for training, set
`settings.mlflow.run_id` with the ID of the run saved and `model_name: final_model`.
Maintain the same data, mapping and model hyperparameters:

```bash
make infer ARGS="--config graph-neural-network/smoke.yaml --sampler full --device cpu"
# After loading customers and products in Neo4j:
make pipeline-recommendations
```

The inference exports `DATA_PATH/rec_tables/10/dataset_res_10_features_True.csv`
with the default configuration: `ids_customer`, `ids_product`, `product_name`.
The path changes with `k` and `add_features`; in that case update also
`RECOMMENDATIONS_CSV_PATH` for the importer Neo4j.

Export uses the test set training context. It's not like a model.
trained on all purchases to serve recommendations in production.

## Data, sampler and ranking

- The graph retains the first purchase of each customer couple–product.
- The time split is 60% training, 20% validation and 20% test, except rounding;
  validation and testing use the training context.
- `neighbor` is the original sampler and requires `pyg-lib` compatible with PyTorch.
- `full` use all the nodes and edges prior to the first event of each batch;
  is suitable for demo and does not replicate sampling `neighbor`.
- Automatic device selection prefers CUDA, then MPS, then CPU.
- The ranking supports `Euclideann` (negative distance), `cosine` and `dot`.
  Excludes context purchases; it explicitly fails if they remain less than
  `k` candidates for a customer.

## Technical documentation

- [Architecture, loss and metrics](../docs/modules/project/network.md)
- [Complete guide of the three components](../docs/modules/getting_started.md)
- [Code Map](../docs/modules/code.md)
- [Verifications and results of training](../docs/verification.md)

HTML is generated from the root with `make docs` and consult with
`make full-docs`. Detailed documentation lives in `docs/`;
collects commands to use the component.
