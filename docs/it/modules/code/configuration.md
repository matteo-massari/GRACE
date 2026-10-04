# Configurazione e tracking

Gli iperparametri sono in `graph-neural-network/interface.yaml`; la demo ridotta
usa `graph-neural-network/smoke.yaml`. `--config` seleziona il file da CLI.
Il blocco `hyperparameters` configura il modello; `settings.mlflow` identifica
la run e il modello da caricare durante l'inferenza.

Il client legge `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT` ed eventuali credenziali
remote da `.env`. Il server locale è gestito da `docker-compose.mlflow.yml`:
backend SQLite e artifact sono nel volume Docker e non richiedono percorsi
locali condivisi con il processo di training.

```{automodule} graph_recc_sys.utils.hyperparams
:members:
:show-inheritance:
```

```{automodule} graph_recc_sys.utils.mlflow_config
:members:
:show-inheritance:
```
