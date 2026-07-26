# Graph Recommendation System

Repository unico per un sistema di raccomandazione basato su grafo. Le dipendenze,
le variabili d'ambiente e i comandi operativi sono condivisi alla radice.

## Componenti

- [`graph-neural-network`](graph-neural-network/): motore GNN per training,
  valutazione e inferenza di raccomandazioni.
- [`graph-rag-agent`](graph-rag-agent/): agente Graph RAG e relative integrazioni.
- [`pipeline`](pipeline/): ETL per il caricamento e l'arricchimento del grafo Neo4j.

## Avvio

```bash
uv sync
cp .envexample .env
make help
```

La configurazione comune vive in `.env`; i comandi disponibili sono centralizzati
nel `Makefile`. Ogni componente conserva soltanto il proprio codice e la propria
documentazione.
