# Internazionalizzazione e dispositivi — 4 ottobre 2026

Dipendenze bloccate installate, test di regressione e build Sphinx rigorose in
entrambe le lingue completati. Training sintetico di un’epoca completato con
selezione automatica della CPU, validazione e metriche finali sul test.
I test coprono anche selezione CUDA/MPS/CPU, errori per dispositivi non disponibili
e compatibilità dei vecchi YAML. CUDA e MPS non sono disponibili in questo runtime:
la selezione è verificata con simulazioni, il training reale solo su CPU.
Neo4j e LLM non sono stati interrogati durante questa modifica.

Le verifiche seguenti appartengono alla revisione precedente e non sono state
tutte ripetute durante questa modifica.

# Verifica locale — 3 ottobre 2026

Ambiente: macOS Apple Silicon, Python 3.12.12, PyTorch 2.11.0,
PyTorch Geometric 2.8.0.post1, MLflow 3.14.0. Architettura GraphSAGE originale
preservata; per la demo si usano dimensioni ridotte e il sampler `full` su CPU.

## Risultati

- Installazione del progetto con `uv sync --locked`: riuscita.
- `make check`: 8 test superati e nessun errore Python nel controllo Ruff mirato.
- `make docs`: build Sphinx riuscita con warning trattati come errori.
- `make smoke-train EPOCHS=10`: training, validazione e test completati.
- Training aggiuntivo di un'epoca con MLflow SQLite locale: checkpoint e modello
  finale salvati, poi ricaricati per l'inferenza.
- Esportazione verificata: 1.000 raccomandazioni, dieci per ciascuno dei 100
  clienti, nessun duplicato cliente–prodotto, identificativi e nomi coerenti con
  le anagrafiche sintetiche.
- Help dell'agente e del loader Neo4j eseguibili senza connessioni; import della
  configurazione agente riuscito.

## Training sintetico di dieci epoche

5.453 eventi generati, ridotti a 2.064 coppie cliente–prodotto distinte. Il grafo
mantiene il primo acquisto della coppia; split: 1.240 training, 412 validation,
412 test. Seed 42 per il training; configurazione `graph-neural-network/smoke.yaml`.

| Metrica finale sul test | Valore |
| --- | ---: |
| Precision@10 | 0,08367 |
| Recall@10 | 0,19346 |
| MAP@10 | 0,06387 |
| Hit ratio@10 | 0,55102 |
| NDCG@10 | 0,13315 |
| Loss training, ultima epoca | 0,4170 |
| Loss validation, ultima epoca | 0,3981 |

È una verifica funzionale: questi numeri non dimostrano qualità predittiva su
clienti reali. Il modello non è stato ottimizzato; la loss non mostra un calo
stabile in questo training breve. Il sampler completo conserva contesti
strettamente precedenti al primo evento di ogni batch, ma non è equivalente al
campionamento dei vicini originale.

## Risultati locali conservati

Gli output pesanti restano fuori da Git:

- `artifacts/verification/training.log`: log delle dieci epoche;
- `artifacts/verification/mlflow.log`: verifica del tracking;
- `artifacts/verification/mlflow.db`: metadati MLflow della verifica;
- `artifacts/verification/inference.yaml`: configurazione del modello verificato;
- `mlruns/`: artifact dei modelli locali;
- `database/gnn/rec_tables/10/dataset_res_10_features_True.csv`: output del modello
  da un'epoca ricaricato da MLflow, distinto dai placeholder.

Per ripetere l'inferenza del modello locale verificato, dalla stessa checkout:

```bash
MLFLOW_TRACKING_URI=sqlite:///artifacts/verification/mlflow.db \
MLFLOW_EXPERIMENT=local-verification DATA_PATH=database/gnn \
uv run python -m graph_recc_sys --no-training --sampler full --device cpu \
  --config artifacts/verification/inference.yaml
```

Questo CSV usa il contesto di training per valutare il test; non è un export
riaddestrato su tutto lo storico per una distribuzione in produzione.

## Limiti delle verifiche

- Alla prima verifica Docker non era attivo. I servizi Docker sono stati poi
  verificati come descritto sotto; caricamento della pipeline e idempotenza sul
  database restano non verificati dal vivo.
- Non sono state inviate richieste LLM: le risposte dell'agente richiedono Neo4j,
  endpoint, credenziali e modelli accessibili.
- `pyg-lib` non è installato: il sampler `neighbor` e CUDA non sono stati eseguiti.
- Impala resta un template da completare con query e driver privati.
- La workflow GitHub Actions è stata aggiunta, ma non eseguita su GitHub.

Il controllo dei file pubblicabili non ha trovato pattern riconoscibili di
chiavi API o chiavi private. `.env` resta ignorato. Questo controllo non è un
audit completo della cronologia Git.

## Verifica dei Compose separati

Nello stesso giorno sono stati introdotti `docker-compose.neo4j.yml` e
`docker-compose.mlflow.yml`, rimuovendo il Compose sperimentale ArcadeDB.

Verifiche eseguite con progetti, container, porte e volumi temporanei separati:

- validazione di entrambi i file con `docker compose config`;
- avvio di Neo4j 5.26.27 e MLflow 3.15.0-full con healthcheck superati;
- connessione Bolt, query Cypher e chiamata `apoc.meta.data()` riuscite;
- creazione di esperimento e run MLflow, registrazione e lettura di una metrica;
- caricamento e download HTTP di un artifact con contenuto verificato.

I container e i volumi di prova sono stati rimossi dopo la verifica. I servizi
preesistenti non sono stati riavviati o modificati. Il nome del volume dati
Neo4j originale è preservato dalla nuova configurazione.

La porta 7474 era occupata da un altro Neo4j: per avviare anche questo progetto,
configurare ad esempio `NEO4J_HTTP_PORT=7475` in `.env`.
