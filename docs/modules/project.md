# The three components

The project distinguishes three responsibilities. Persistent services are Neo4j and
MLflow, each with a Compose; pipeline, training and agent are Python programs
Start when you need them.

| Component | Input | Responsibility | Output |
| --- | --- | --- | --- |
| Knowledge Graph and pipeline | CSV of customers, products, sellers and purchases | Build and update your business domain in Neo4j | Nodes, relationships, and recommendations |
| GNN | CSV in format `DATA_PATH/training_tables/` | Build the customer graph–product, train and evaluate GraphSAGE | Models/metrics in MLflow and CSV of recommendations |
| Graph RAG | Question, Neo4j schema and Cypher results | Recover evidence from the graph and formulate an answer | Text reply in the user’s language (English by default) |

## Flow implemented

```{mermaid}
flowchart TD
    D["CSV data"] --> P["Pipeline Neo4j"]
    P --> K["Knowledge Graph / Neo4j"]
    D --> G["PyG graph and GNN training"]
    G --> M["MLflow: models and metrics"]
    M --> I["GNN inference"]
    D --> I
    I --> C["Recommendation CSV"]
    C --> R["Import recommendations"]
    R --> K
    U["User question"] --> A["Graph RAG + LLM"]
    A -->|Cypher| K
    K -->|Results| A
    A --> O["Evidence-based answer"]
```

The GNN **reads CSV files**. The agent **reads recommendations imported into Neo4j**;
it does not invoke the GNN or access MLflow.
Synthetic data generation feeds both paths with the same
customer base and products.

## Execution

- `docker-compose.neo4j.yml`: database and APOC Core; start with `make start-neo4j`.
- `docker-compose.mlflow.yml`: tracking, SQLite and artifact; start with `make start-mlflow`.
- `make pipeline-synthetic`: Loading synthetic data in Knowledge Graph.
- `make train` / `make infer`: training and inference of GNN.
- `make pipeline-recommendations`: import of export GNN to Neo4j.
- `make agent ARGS="..."`: a question to the terminal agent.

The demo `make smoke-train` check the network without external services. The guide
operational describes the[full start](getting_started.md), including credentials and
passing the MLflow run to inference.

```{toctree}
:maxdepth: 1

Knowledge Graph and pipeline <project/graph_doc>
Neural Network GNN <project/network>
Graph RAG <project/agent_doc>
```
