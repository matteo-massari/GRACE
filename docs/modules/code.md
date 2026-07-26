# Architettura del codice

```{toctree}
:maxdepth: 3
:caption: Documentation

engine <code/engine>
extraction <code/extraction>
graph_dataset <code/graph_dataset>
graph_loader <code/graph_loader>
neural_network <code/neural_network>
query <code/query>
```

Documentazione del **Graph Recommendation System**, una pipeline end-to-end per costruire, addestrare, valutare e utilizzare un sistema di raccomandazione neurale basato su **grafi eterogenei** con nodi di tipo **utente** e **prodotto**.

## Componenti principali

I blocchi fondamentali del sistema sono:

- `DatasetManager`: gestione dell’estrazione dati e della persistenza locale;
- `MarketBasketGraphBuilder`: costruzione del grafo eterogeneo user-item;
- `GraphLoader`: creazione degli split temporali e dei loader;
- `RecommenderModel`: modello neurale basato su GraphSAGE;
- `Engine`: Orchetratore di infereza, training e Valutazione;
- `Training`, `Inference`, `Evaluation`: componenti specializzati della pipeline;
- `NeuralNetworkEngine`: orchestrazione di training, validazione, test e inferenza;
- `MlflowConfig`: integrazione con MLflow per tracking e caricamento modelli.

## Flusso generale

Il funzionamento della pipeline segue questi passaggi:

1. estrazione e salvataggio dei dati;
2. costruzione del grafo eterogeneo;
3. creazione dei sottografi e dei loader;
4. inizializzazione del modello;
5. training oppure caricamento di un modello già salvato;
6. inferenza e valutazione;
7. esportazione delle raccomandazioni finali.



