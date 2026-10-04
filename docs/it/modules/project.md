# Le tre componenti

Il progetto distingue tre responsabilità. I servizi persistenti sono Neo4j e
MLflow, ciascuno con un Compose; pipeline, training e agente sono programmi Python
avviati quando servono.

| Componente | Input | Responsabilità | Output |
| --- | --- | --- | --- |
| Knowledge Graph e pipeline | CSV di clienti, prodotti, venditori e acquisti | Costruire e aggiornare il dominio commerciale in Neo4j | Nodi, relazioni e raccomandazioni consultabili |
| GNN | CSV nel formato `DATA_PATH/training_tables/` | Costruire il grafo cliente–prodotto, addestrare e valutare GraphSAGE | Modelli/metriche in MLflow e CSV delle raccomandazioni |
| Graph RAG | Domanda, schema Neo4j e risultati Cypher | Recuperare evidenze dal grafo e formulare una risposta | Risposta testuale in italiano |

## Flusso implementato

```{mermaid}
flowchart TD
    D["Dati CSV"] --> P["Pipeline Neo4j"]
    P --> K["Knowledge Graph / Neo4j"]
    D --> G["Grafo PyG e training GNN"]
    G --> M["MLflow: modelli e metriche"]
    M --> I["Inferenza GNN"]
    D --> I
    I --> C["CSV raccomandazioni"]
    C --> R["Importazione raccomandazioni"]
    R --> K
    U["Domanda utente"] --> A["Graph RAG + LLM"]
    A -->|Cypher| K
    K -->|Risultati| A
    A --> O["Risposta con evidenze"]
```

La GNN **legge i CSV**, non Neo4j. L'agente **legge le raccomandazioni importate
nel grafo**: non invoca direttamente il modello GNN e non accede a MLflow.
La generazione dei dati sintetici alimenta entrambi i percorsi con la stessa
base di clienti e prodotti.

## Esecuzione

- `docker-compose.neo4j.yml`: database e APOC Core; avvio con `make start-neo4j`.
- `docker-compose.mlflow.yml`: tracking, SQLite e artifact; avvio con `make start-mlflow`.
- `make pipeline-synthetic`: caricamento dei dati sintetici nel Knowledge Graph.
- `make train` / `make infer`: addestramento e inferenza della GNN.
- `make pipeline-recommendations`: importazione dell'export GNN in Neo4j.
- `make agent ARGS="..."`: una domanda all'agente da terminale.

La demo `make smoke-train` verifica la sola rete senza servizi esterni. La guida
operativa descrive l'[avvio completo](getting_started.md), inclusi credenziali e
passaggio della run MLflow all'inferenza.

```{toctree}
:maxdepth: 1

Knowledge Graph e pipeline <project/graph_doc>
Rete neurale GNN <project/network>
Graph RAG <project/agent_doc>
```
