# Graph Recommendation System
[English version](README.md)

Sistema di raccomandazione con una GNN eterogenea cliente–prodotto, una pipeline
Neo4j e un assistente Graph RAG. Il modello usa GraphSAGE, split temporali e
valutazione top-k; il dataset dimostrativo è interamente sintetico.

## Tre componenti, due servizi Docker

| Componente | Ruolo |
| --- | --- |
| Knowledge Graph e pipeline | Caricano CSV e raccomandazioni in Neo4j |
| Rete GNN | Legge CSV, costruisce il grafo PyG e produce raccomandazioni |
| Graph RAG | Interroga Neo4j con Cypher e risponde tramite LLM |

Il flusso è **CSV → GNN → CSV raccomandazioni → pipeline → Neo4j → Graph RAG**.
Le anagrafiche e gli acquisti alimentano anche Neo4j direttamente tramite la
pipeline. La GNN non legge Neo4j; l'agente non esegue direttamente la rete.
MLflow traccia training e modelli. Neo4j e MLflow sono i due servizi Docker;
le tre componenti Python si avviano con i comandi riportati sotto.

## Avvio rapido

Richiede Python 3.12 e `uv`. Eseguire i comandi dalla radice della repository:

```bash
uv sync --locked
cp .envexample .env  # solo al primo avvio, senza sovrascrivere la propria configurazione
make smoke-train EPOCHS=10
```

La demo genera 100 clienti, 60 prodotti, 30 venditori e 5.453 acquisti, quindi
esegue training, validazione e test sul dispositivo selezionato automaticamente. Non richiede Neo4j, credenziali LLM,
Impala, CUDA o un server MLflow. `--dry-run` disattiva il tracking: **il training
viene comunque eseguito**. I risultati misurano il funzionamento della pipeline,
non la qualità di un modello pronto per produzione.

## Struttura

| Percorso | Contenuto |
| --- | --- |
| `graph-neural-network/src/graph_recc_sys/` | Modello, dataset, training, inferenza e metriche |
| `graph-neural-network/interface.yaml` | Configurazione originale del modello |
| `graph-neural-network/smoke.yaml` | Configurazione ridotta della demo |
| `graph-rag-agent/src/graph_rag_agent/` | Assistente e flusso di interrogazione Neo4j |
| `pipeline/` | Generazione dati, caricamento Neo4j e template ETL aziendale |
| `database/` | Dataset generati localmente, esclusi da Git |
| `docs/` | Unica documentazione Sphinx, teoria e storico esperimenti |
| `tests/` | Controlli su dati, temporalità, ranking ed esportazione |

## Servizi Docker

Due Compose indipendenti:

| File | Servizio | Accesso predefinito |
| --- | --- | --- |
| `docker-compose.neo4j.yml` | Knowledge Graph Neo4j con APOC Core | `http://127.0.0.1:7474`, Bolt `localhost:7687` |
| `docker-compose.mlflow.yml` | Tracking MLflow, database SQLite e artifact | `http://127.0.0.1:1234` |

```bash
make start-neo4j
make start-mlflow
# Arresto indipendente, senza eliminare i dati:
make stop-neo4j
make stop-mlflow
```

I dati persistono nei volumi Docker. Neo4j mantiene il nome del progetto e il
volume dati originali; il Compose MLflow usa un progetto separato. Gli artifact
MLflow vengono caricati e scaricati tramite il server HTTP, anche quando il
training gira fuori da Docker. I precedenti esperimenti SQLite sul computer non
vengono trasferiti automaticamente nel nuovo volume Docker.

Le porte si configurano con `NEO4J_HTTP_PORT`, `NEO4J_BOLT_PORT` e `MLFLOW_PORT`.
Se 7474 è già occupata da un altro Neo4j, impostare ad esempio
`NEO4J_HTTP_PORT=7475` in `.env`. Se cambia la porta Bolt o MLflow, aggiornare
anche `NEO4J_URI` o `MLFLOW_TRACKING_URI`. L'utente iniziale Neo4j è `neo4j`;
la password in `.env` deve coincidere con quella del database già esistente.

GNN, pipeline e Graph RAG continuano a essere eseguiti con Python/Make.
`make local-mlflow-up` e `mlflow_init.sh` restano alias per l'avvio Docker.

## Training e inferenza

```bash
make smoke-train                 # tre epoche, dispositivo automatico, dati sintetici rigenerati
make dry-run ARGS="--sampler full --epochs 5"  # dimensioni originali del modello
make start-mlflow                # server Docker, configurato tramite .env
make train ARGS="--sampler full --epochs 10"
make infer ARGS="--sampler full"
```

Per l'inferenza, impostare `settings.mlflow.run_id` e `model_name` in
`graph-neural-network/interface.yaml`, con gli stessi iperparametri usati nel
training. La demo ridotta richiede `--config graph-neural-network/smoke.yaml`
anche durante l'inferenza. Gli output vanno in `DATA_PATH/rec_tables/`.

Il sampler predefinito `neighbor` conserva il percorso originale e richiede
`pyg-lib` compatibile con la versione di PyTorch. `--sampler full` è un'alternativa
esplicita per piccoli dataset: mantiene tutti i nodi e usa solo archi precedenti
al primo evento del batch. Consuma più memoria sui grafi grandi e non riproduce
lo stesso campionamento del percorso originale. La selezione automatica usa CUDA, poi MPS, infine CPU. Per forzare il dispositivo: `--device cpu`.

Il grafo conserva il primo acquisto di ciascuna coppia cliente–prodotto:
l'obiettivo è raccomandare prodotti nuovi. Le cache vengono ricostruite quando
cambiano i CSV sorgente. Le raccomandazioni placeholder sono sempre separate
dall'output GNN e marcate `placeholder_not_gnn`.

## Neo4j e agente

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
make agent ARGS="Quali sono i clienti di Roma con più acquisti?"
```

Neo4j richiede Docker attivo e le variabili `NEO4J_*` in `.env`. L'agente richiede
anche `AZURE_OPENAI_API_KEY` e `AZURE_OPENAI_ENDPOINT`, con accesso ai modelli
configurati nel codice. Il comando esegue una domanda da terminale, senza
avviare un server web. Vedere [pipeline](pipeline/README.md) e
[agente](graph-rag-agent/README.md).

`pipeline/ETL_fact.py` conserva il template originale: le query Impala e il
connettore privato non sono inclusi e il comando segnala esplicitamente questa
limitazione. Per la demo usare `pipeline-synthetic`.

## Verifiche e documentazione

```bash
make check       # test di regressione e controllo degli errori Python
make docs        # genera docs/_build/html/index.html
make full-docs   # documentazione bilingue su localhost:8000
make help
```

La [guida operativa completa](docs/it/modules/getting_started.md) segue il percorso
fino all'importazione delle raccomandazioni e alla domanda all'agente.
La [documentazione tecnica](docs/it/index.md) approfondisce
[Knowledge Graph](docs/it/modules/project/graph_doc.md),
[GNN](docs/it/modules/project/network.md) e [Graph RAG](docs/it/modules/project/agent_doc.md).
I README sono guide rapide; gli HTML vengono generati dagli stessi sorgenti
tecnici in `docs/`. Lo storico degli esperimenti è distinto dalle istruzioni attuali.

Il [resoconto delle verifiche](docs/it/verification.md) distingue ciò che è stato
eseguito dalle integrazioni che richiedono servizi esterni. La workflow GitHub Actions è configurata per eseguire
i controlli, un training breve e la build della documentazione. `.env`, dataset rigenerabili, modelli, cache e
HTML generato sono esclusi dal repository.

Licenza: [GPL-3.0](LICENSE.txt).
