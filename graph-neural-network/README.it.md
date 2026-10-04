# Rete neurale di raccomandazione
[English version](README.md)

La GNN apprende le relazioni cliente–prodotto con GraphSAGE e produce
raccomandazioni top-k. Legge CSV locali e costruisce un grafo PyTorch Geometric:
non legge direttamente il Knowledge Graph Neo4j.

## Demo locale

Dalla radice della repository, dopo `uv sync --locked`:

```bash
make smoke-train EPOCHS=10
```

Il comando rigenera i dati sintetici e completa training, validazione e test su
CUDA/MPS/CPU in automatico, senza MLflow o altri servizi. `--dry-run` significa training senza tracking,
non simulazione. La demo usa `smoke.yaml` (dimensioni 32/16/8); `interface.yaml`
conserva la configurazione originale (512/256/128, 50 epoche).

## Training con MLflow

Configurare `.env` dall'esempio alla radice, con `DATA_PATH=database/gnn` e
`MLFLOW_TRACKING_URI` coerente con la porta del servizio:

```bash
make synthetic-data
make start-mlflow
make train ARGS="--config graph-neural-network/smoke.yaml --sampler full --device cpu --epochs 10"
```

MLflow salva checkpoint, modello finale, iperparametri e metriche. Il server
Docker gestisce database e artifact nel proprio volume; il client usa HTTP.
Per le dimensioni originali, omettere `--config`.

## Inferenza e importazione

Nella configurazione YAML usata per il training, impostare
`settings.mlflow.run_id` con l'ID della run salvata e `model_name: final_model`.
Mantenere gli stessi dati, mapping e iperparametri del modello:

```bash
make infer ARGS="--config graph-neural-network/smoke.yaml --sampler full --device cpu"
# Dopo aver caricato clienti e prodotti in Neo4j:
make pipeline-recommendations
```

L'inferenza esporta `DATA_PATH/rec_tables/10/dataset_res_10_features_True.csv`
con la configurazione predefinita: `ids_customer`, `ids_product`, `product_name`.
Il percorso cambia con `k` e `add_features`; in quel caso aggiornare anche
`RECOMMENDATIONS_CSV_PATH` per l'importatore Neo4j.

L'export usa il contesto di training del test set. Non equivale a un modello
riaddestrato su tutti gli acquisti per servire raccomandazioni in produzione.

## Dati, sampler e ranking

- Il grafo conserva il primo acquisto di ogni coppia cliente–prodotto.
- Lo split temporale è 60% training, 20% validation e 20% test, salvo arrotondamenti;
  validation e test usano il contesto di training.
- `neighbor` è il sampler originale e richiede `pyg-lib` compatibile con PyTorch.
- `full` usa tutti i nodi e gli archi precedenti al primo evento di ogni batch;
  è adatto alla demo e non replica il campionamento di `neighbor`.
- Il dispositivo automatico è CUDA, poi MPS, infine CPU.
- Il ranking supporta `euclidean` (distanza negativa), `cosine` e `dot`.
  Esclude gli acquisti del contesto; fallisce esplicitamente se restano meno di
  `k` candidati per un cliente.

## Documentazione tecnica

- [Architettura, loss e metriche](../docs/it/modules/project/network.md)
- [Guida completa delle tre componenti](../docs/it/modules/getting_started.md)
- [Mappa del codice](../docs/it/modules/code.md)
- [Verifiche e risultati del training](../docs/it/verification.md)

Gli HTML si generano dalla radice con `make docs` e si consultano con
`make full-docs`. La documentazione dettagliata vive in `docs/`; questo README
raccoglie i comandi per usare il componente.
