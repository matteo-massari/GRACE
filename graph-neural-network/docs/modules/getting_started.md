# Getting Started
Questa guida spiega come configurare l’ambiente ed eseguire la pipeline del motore di raccomandazione GNN in locale per la prima volta.

Affronteremo come:
- preparare l’ambiente Python;
- configurare i file necessari;
- impostare MLflow;
- avviare il training o l’inferenza;
- verificare che la rete sia partita correttamente.

Prima di iniziare, assicurarsi di avere `uv` installato;


## 1. Preparare l’ambiente

Creare o usare l’ambiente Python del progetto e installare le dipendenze definite nel repository in particolare nel `project.toml` con:

```bash
uv sync
```

Per verificare che l’ambiente sia pronto:
```bash
uv run python3 --version
```

## 2. Configurare le variabili d’ambiente

Il progetto usa variabili d’ambiente per la configurazione dei dati e del tracking esperimenti.

È consigliato creare un file `.env` nella root del progetto, un esempio è il `.envexample`.

Esempio minimo:

```text
MLFLOW_BACKEND_URI=sqlite:////path/mlflow.db
MLFLOW_ARTIFACT_ROOT=file:///path/artifacts
MLFLOW_HOST=127.0.0.1
MLFLOW_PORT=1234
MLFLOW_TRACKING_URI=http://000.0.0.0:1234
MLFLOW_EXPERIMENT=NomeEsperimento
DATA_PATH=/path/to/data
```

### Note sulle variabili MLflow

- `MLFLOW_BACKEND_URI`: backend store dei metadati MLflow;
- `MLFLOW_ARTIFACT_ROOT`: directory locale degli artifact;
- `MLFLOW_HOST`: host del server MLflow;
- `MLFLOW_PORT`: porta del server MLflow;
- `MLFLOW_TRACKING_URI`: endpoint usato dal client per loggare esperimenti;
- `MLFLOW_EXPERIMENT`: nome dell’esperimento in cui tracciare le run.

Se si usa un’istanza remota di MLflow, è possibile configurare anche:

```text
MLFLOW_TRACKING_URI=https://istanza-mlflow
MLFLOW_TRACKING_USERNAME=username
MLFLOW_TRACKING_PASSWORD=password
MLFLOW_TRACKING_INSECURE_TLS=true
```
Al'interno in base alle variabili presenmti nell'env, il codice deciderà in autoamtico che istanza usare

## 3. Configurare `interface.yaml`

Il file `interface.yaml` deve contenere almeno:

- configurazione del training;
- configurazione del modello;
- parametri del grafo;
- parametri dei loader;
- parametri di valutazione.

In particolare, verificare che siano valorizzati i parametri fondamentali come:

- modalità del modello (`binary` oppure `triplet`);
- dimensioni degli embedding;
- numero di layer;
- learning rate;
- numero di epoche;
- top-k di valutazione;
- parametri di split temporale;
- parametri del neighborhood sampling.

Se si usa la modalità di inferenza con caricamento da MLflow, assicurarsi che la configurazione sia coerente con il modello salvato.

## 4. Avviare MLflow in locale

Se si vuole tracciare gli esperimenti in locale, avviare prima il server MLflow, con l'apposito comando

```text
make mlflow-local
```

Dopo l’avvio, verificare che il server risponda all’indirizzo configurato, ad esempio:

```text
http://127.0.0.1:1234
```

Aprendo l’interfaccia web dovrebbe essere visibile la UI di MLflow.

## 5. Preparare i dati
La pipeline gestisce automaticamente il caricamento dei dati tramite la classe `DatasetManager`. In base alla configurazione definita nel file `interface.yaml`, è possibile:

- scaricare nuovi dati da Impala;
- riutilizzare file CSV già presenti nelle directory previste.

Prima dell’esecuzione, è importante verificare che:

- il percorso dei dati sia configurato correttamente;
- le directory locali siano accessibili in scrittura;
- i file richiesti siano già disponibili, oppure che le query verso Impala siano eseguibili correttamente.

Se i dati non sono disponibili in locale, la pipeline tenterà automaticamente di recuperarli e salvarli nei percorsi configurati.

## 6. Eseguire il training

Per eseguire il training completo con tracking MLflow attivo:

```bash
uv run python3 -m graph_recc_sys
```

Questo comando esegue il flusso completo:

1. caricamento configurazione;
2. inizializzazione MLflow;
3. preparazione dei dataset;
4. costruzione del grafo eterogeneo;
5. creazione degli split temporali;
6. inizializzazione del modello;
7. training;
8. valutazione;
9. salvataggio artifact e metriche.

## 7. Eseguire la pipeline senza MLflow

Per eseguire la pipeline senza logging su MLflow:

```bash
uv run python3 -m graph_recc_sys --dry-run
```

Questa modalità è utile per verificare che il flusso end-to-end funzioni anche senza tracking degli esperimenti.

## 8. Eseguire inferenza da modello già salvato

Per saltare il training e caricare un modello da MLflow:

```bash
uv run python3 -m graph_recc_sys --no-training
```

Questa modalità richiede che:

- la configurazione MLflow sia corretta;
- il modello richiesto sia disponibile nel tracking server;
- la configurazione del grafo sia compatibile con quella usata in training.

## 9. Come capire se la rete è partita correttamente

Una prima esecuzione può essere considerata riuscita se si osservano i seguenti segnali:

- il file di configurazione viene caricato senza errori;
- i dataset vengono trovati o scaricati correttamente;
- il grafo eterogeneo viene costruito e serializzato;
- i loader di training e validation vengono creati;
- il modello viene inizializzato senza mismatch dimensionali;
- in training compaiono loss di train e validation;
- in MLflow compaiono run, metriche e artifact;
- al termine vengono prodotti file di output, come raccomandazioni e metriche finali.

## 10. Output attesi

Durante o dopo l’esecuzione, il progetto può produrre:

- dataset grezzi persistiti in locale;
- grafo processato per PyTorch Geometric;
- mapping utente-prodotto;
- checkpoint del modello;
- metriche di validation e test;
- artifact MLflow;
- file CSV con raccomandazioni finali.

## Troubleshooting

### MLflow non parte

Verificare che:

- il path del backend SQLite sia valido;
- la directory degli artifact esista o sia creabile;
- host e porta non siano già occupati.

### Il training fallisce all’avvio

Verificare che:

- `interface.yaml` sia coerente;
- le feature attese dal modello siano presenti nei dati;
- le dimensioni configurate siano compatibili con il grafo costruito;
- le dipendenze PyTorch e PyTorch Geometric siano installate correttamente.

### Nessun dato disponibile

Verificare che:

- `DATA_PATH` punti alla directory corretta;
- i CSV siano presenti;
- le query del `DatasetManager` possano essere eseguite nel contesto locale.

### L’inferenza da MLflow non trova il modello

Verificare che:

- `MLFLOW_TRACKING_URI` sia corretto;
- l’esperimento esista;
- il modello o la run richiesta siano disponibili;
- la configurazione corrente sia compatibile con il modello tracciato.

## Esempio di flusso minimo

Per una prima prova in locale:

1. configurare `.env`;
2. configurare `interface.yaml`;
3. avviare MLflow;
4. lanciare il training.

Comandi:

```bash
make mlflow
uv run python3 -m graph_recc_sys
```

Se l’obiettivo è solo verificare che la pipeline parta, si può usare:

```bash
uv run python3 -m graph_recc_sys --dry-run
```

## Prossimi passi

Dopo la prima esecuzione riuscita, è consigliato:

- verificare le metriche di validation e test;
- confrontare più run in MLflow;
- controllare la qualità delle raccomandazioni generate;
- consolidare la configurazione in file versionati per ambienti diversi.