# Avvio e configurazione

Tutti i comandi si eseguono dalla **radice della repository**. Servono Python
3.12 e `uv`; Docker è necessario per Neo4j e MLflow. La demo della sola rete
non richiede Docker o credenziali LLM.

## 1. Preparare l'ambiente

```bash
uv sync --locked
cp .envexample .env
```

Copiare `.envexample` soltanto al primo avvio: non sovrascrivere una configurazione
esistente. I dataset sono sintetici e rigenerabili; `.env`, cache, CSV generati,
modelli e HTML compilati restano fuori da Git.

## 2. Provare la rete senza servizi

```bash
make smoke-train EPOCHS=10
```

Genera i dati e completa training, validation e test sul dispositivo selezionato automaticamente con `smoke.yaml` e
sampler `full`. Le dimensioni sono ridotte; `--dry-run` disattiva solo MLflow.
La demo non salva un modello MLflow e non esporta il CSV delle raccomandazioni:
per questi passaggi proseguire con training tracciato e inferenza.

## 3. Configurare e avviare i servizi

| Variabile | Valore dell'esempio / significato |
| --- | --- |
| `DATA_PATH` | `database/gnn`, radice dei dati GNN |
| `NEO4J_HTTP_PORT` | `7474`, porta browser sul computer |
| `NEO4J_BOLT_PORT` | `7687`, porta driver sul computer |
| `NEO4J_URI` | `bolt://localhost:7687`, deve seguire la porta Bolt |
| `NEO4J_USERNAME` | `neo4j`, utente iniziale del database |
| `NEO4J_PASSWORD` | Password iniziale; su volume esistente deve coincidere con quella già impostata |
| `NEO4J_DATABASE` | `neo4j` |
| `MLFLOW_PORT` | `1234`, porta HTTP del tracking server sul computer |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:1234`, deve seguire `MLFLOW_PORT` |
| `MLFLOW_EXPERIMENT` | Nome dell'esperimento del client |
| `AZURE_OPENAI_API_KEY` | Credenziale LLM, necessaria solo per l'agente |
| `AZURE_OPENAI_ENDPOINT` | Base URL del servizio LLM compatibile |
| `RECOMMENDATIONS_CSV_PATH` | Opzionale: percorso dell'export da importare in Neo4j |

```bash
make start-neo4j
make start-mlflow
```

I due Compose sono `docker-compose.neo4j.yml` e `docker-compose.mlflow.yml`.
Partono in background e attendono i rispettivi healthcheck. Con le porte
predefinite, Neo4j Browser è su `http://127.0.0.1:7474` e MLflow su
`http://127.0.0.1:1234`.

Se 7474 è occupata, impostare ad esempio `NEO4J_HTTP_PORT=7475`. Modificare la
porta HTTP Neo4j non cambia la connessione Bolt; modificando la porta Bolt o
MLflow occorre invece aggiornare anche il corrispondente URI client.

Neo4j conserva dati e log nei suoi volumi. MLflow usa SQLite e una directory
artifact nel proprio volume, serviti via HTTP al processo Python. I precedenti
esperimenti SQLite locali non vengono migrati automaticamente. Le vecchie
variabili `MLFLOW_BACKEND_URI`, `MLFLOW_ARTIFACT_ROOT` e `MLFLOW_HOST` non sono
usate dai Compose attuali: i percorsi interni e il bind del server sono definiti
nel file MLflow.

## 4. Caricare il Knowledge Graph

```bash
make synthetic-data
make pipeline-synthetic
```

Sono caricati clienti, prodotti, venditori, gerarchie e acquisti. Il percorso
standard non importa raccomandazioni placeholder. Il codice Impala privato non
serve per questa demo.

## 5. Addestrare e salvare il modello

Esempio completo con la rete ridotta:

```bash
make train ARGS="--config graph-neural-network/smoke.yaml --sampler full --device cpu --epochs 10"
```

MLflow registra iperparametri, checkpoint di epoca e `final_model`, oltre a loss
e metriche. Per usare le dimensioni originali omettere `--config`: il file
predefinito è `graph-neural-network/interface.yaml`.

Il sampler `neighbor` predefinito richiede `pyg-lib` compatibile con PyTorch.
`full` mantiene tutti i nodi e usa soltanto archi precedenti al primo target del
batch: funziona senza quell'estensione ma è adatto a piccoli dataset.

```bash
uv run python -m graph_recc_sys --help
```

Le altre opzioni principali sono `--epochs`, `--seed`, `--device`, `--dry-run` e
`--no-training`. La selezione automatica del dispositivo usa CUDA, poi MPS, infine CPU.

## 6. Generare e importare le raccomandazioni

Dall'interfaccia MLflow copiare l'ID della run appena conclusa. Nel file YAML
usato per il training impostare:

```yaml
settings:
  mlflow:
    run_id: "ID_DELLA_RUN_SALVATA"
    model_name: "final_model"
```

Modificare il blocco esistente senza rimuovere `hyperparameters`. Conservare
modello, dati e mapping coerenti con la run; il nome dell'esperimento del client
è letto da `MLFLOW_EXPERIMENT`, non da `settings.mlflow.experiment_name`.

```bash
make infer ARGS="--config graph-neural-network/smoke.yaml --sampler full --device cpu"
make pipeline-recommendations
```

Con i parametri della demo, il CSV è
`database/gnn/rec_tables/10/dataset_res_10_features_True.csv` e contiene
`ids_customer`, `ids_product`, `product_name`. Se cambiano `DATA_PATH`, `k` o
`add_features`, indicare il nuovo file tramite `RECOMMENDATIONS_CSV_PATH`.

L'export è prodotto sul contesto del test. L'importatore aggiorna le relazioni
esistenti ma non elimina automaticamente raccomandazioni di run precedenti.
I [limiti della pipeline](project/graph_doc.md) descrivono anche i placeholder.

## 7. Interrogare Graph RAG

Configurare le credenziali LLM e verificare che l'endpoint esponga i modelli
indicati nel [capitolo dell'agente](project/agent_doc.md), poi:

```bash
make agent ARGS="Quali sono i clienti di Roma con più acquisti?"
```

Il comando esegue una domanda da terminale e termina dopo la risposta. Non avvia
un server web. Per le domande sulle raccomandazioni GNN, completare prima il
passaggio 6. `make agent ARGS="--help"` mostra l'uso senza collegarsi ai servizi.

## Controlli, documentazione e arresto

```bash
make check
make docs
make full-docs
```

`make docs` genera `docs/_build/html/index.html`, con warning Sphinx trattati
come errori. `make full-docs` avvia la consultazione locale in entrambe le lingue; ricompilare dopo ogni modifica e interromperla con Ctrl+C. Gli HTML sono derivati dai sorgenti e non
vanno modificati direttamente.

```bash
make stop-neo4j
make stop-mlflow
```

L'arresto è indipendente e conserva i volumi. `make local-mlflow-up` e
`mlflow_init.sh` sono alias dell'avvio Docker MLflow. I risultati effettivamente
verificati sono raccolti nel [resoconto locale](../verification.md).
