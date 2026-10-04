# Neo4j ETL Pipeline
[English version](README.md)

Per la demo senza Impala:

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
```

I CSV di `database/` sono fittizi, deterministici e rigenerabili. Il loader
usa `MERGE` per evitare duplicati e aggrega gli eventi di acquisto sulle relazioni cliente–prodotto.

`make pipeline-recommendations` importa l'output GNN da
`database/gnn/rec_tables/10/dataset_res_10_features_True.csv`, dopo l'inferenza.
È possibile indicare un altro CSV tramite `RECOMMENDATIONS_CSV_PATH` in `.env`.
I placeholder sono separati e si caricano solo esplicitamente:

```bash
uv run python pipeline/load_synthetic_neo4j.py --include-placeholder-recommendations
```

`ETL_fact.py` conserva il template aziendale originale, con SQL e connessione
Impala da completare. Segnala questa limitazione prima di aprire una connessione.
La demo usa invece `load_synthetic_neo4j.py`.

## Collegamento con la rete

La GNN legge i CSV sotto `database/gnn/`; non estrae il grafo da Neo4j.
Dopo il training tracciato e l'inferenza, `ETL_recc.py` collega i codici
`ids_customer` e `ids_product` dell'export a clienti e prodotti `Level5` già
caricati. Se cambiano `DATA_PATH`, `k` o `add_features`, aggiornare
`RECOMMENDATIONS_CSV_PATH`.

I loader inseriscono o aggiornano nodi e relazioni; non rimuovono automaticamente
quelli assenti dal nuovo input. In particolare, una nuova importazione non
sostituisce tutte le raccomandazioni precedenti e non ripulisce gli attributi dei
placeholder. Tenere distinta una prova con placeholder dall'uso dei risultati GNN.

- [Schema e comportamento delle pipeline](../docs/it/modules/project/graph_doc.md)
- [Sequenza completa: dati, training, import e agente](../docs/it/modules/getting_started.md)
