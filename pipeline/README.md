# Neo4j ETL pipeline

**English** | [Italiano](README.it.md)

Run the demo without Impala from the repository root:

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
```

CSV files in `database/` contain deterministic synthetic data. The loader uses
`MERGE` to avoid duplicates and aggregates purchase events into customer–product
relationships. Geographic names and legacy source field `partita_iva` (VAT number)
retain their source spelling for compatibility; demo descriptions are English.

After GNN inference, `make pipeline-recommendations` imports
`database/gnn/rec_tables/10/dataset_res_10_features_True.csv`. Override
`RECOMMENDATIONS_CSV_PATH` in `.env` for a different path, including when changing
`DATA_PATH`, `k`, or `add_features`.

Placeholder recommendations are separate and imported only when explicitly requested:

```bash
uv run python pipeline/load_synthetic_neo4j.py --include-placeholder-recommendations
```

`ETL_fact.py` preserves the original integration template. Private SQL queries and
the Impala connector must be supplied before use; it reports this requirement
before connecting. The demo uses `load_synthetic_neo4j.py` instead.

The GNN reads local CSV files. `ETL_recc.py` connects exported `ids_customer` and
`ids_product` values to existing customers and Level5 products in Neo4j.
Loaders upsert nodes and relationships; they do not remove records absent from new
input, replace all earlier recommendations, or clear placeholder attributes.
Use a separate demo database when testing placeholder recommendations.

- [Graph schema and pipeline behavior](../docs/modules/project/graph_doc.md)
- [End-to-end guide](../docs/modules/getting_started.md)
