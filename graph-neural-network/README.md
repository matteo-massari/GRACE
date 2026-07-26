# Motore di Raccomandazione GNN per Grafi Eterogenei User-Item

## Panoramica

Questo progetto implementa una pipeline end-to-end per costruire, addestrare, valutare e utilizzare un sistema di raccomandazione neurale basato su **grafi eterogenei** con nodi di tipo **utente** e **prodotto**.

La codebase copre l’intero flusso operativo:

- estrazione dei dati da sorgenti esterne tramite query;
- salvataggio locale dei dataset e delle tabelle di supporto;
- costruzione di un grafo eterogeneo compatibile con **PyTorch Geometric**;
- creazione di split temporali train/validation/test;
- addestramento di un modello GNN per **link prediction**;
- valutazione con metriche di ranking top-k;
- inferenza finale con MIPS search.

L’obiettivo del progetto è raccomandare, per ciascun cliente, i prodotti più rilevanti sulla base delle interazioni storiche e delle eventuali feature descrittive di utenti e articoli.

---

## Obiettivo del sistema

Il motore apprende rappresentazioni latenti di:

- **clienti**;
- **prodotti**;
- **relazioni di acquisto** tra clienti e prodotti.

A partire da tali rappresentazioni, il sistema genera raccomandazioni **top-k** per ciascun utente, escludendo gli item già osservati nel contesto noto ed eventualmente confrontando le predizioni con il ground truth di validation o test.

Il progetto supporta due paradigmi principali di apprendimento:

- **modalità binary**: il modello produce uno score per ogni arco candidato user-item;
- **modalità triplet/contrastive**: il modello produce embedding normalizzati utilizzabili con loss metriche o contrastive.

---

## Architettura logica

La pipeline è composta dai seguenti blocchi principali:

1. **DatasetManager**  
   Scarica e organizza i dataset grezzi e le tabelle ausiliarie.

2. **MarketBasketGraphBuilder**  
   Legge i CSV, pulisce i dati e costruisce il grafo eterogeneo user-item.

3. **GraphLoader**  
   Esegue split temporali e crea i `LinkNeighborLoader` per training e validation.

4. **RecommenderModel**  
   Modello GNN eterogeneo basato su GraphSAGE per link prediction / metric learning.

5. **Training / Inference / Evaluation**  
   Componenti specializzati per addestramento, inferenza e valutazione.

6. **NeuralNetworkEngine**  
   Orchestratore ad alto livello dell’intero ciclo di training, validazione, test e inferenza.

7. **MlflowConfig**  
   Gestione del tracking degli esperimenti e caricamento dei modelli salvati.

8. **Script principale della pipeline**  
   Punto di ingresso che connette tutti i componenti e permette l’esecuzione da riga di comando.

---

## Struttura funzionale del progetto

### 1. Gestione dati e persistenza

La classe `DatasetManager` si occupa di:

- leggere la configurazione da variabili d’ambiente;
- costruire i path locali per dataset, recommendation tables e analysis tables;
- eseguire query verso la sorgente dati;
- salvare i risultati in formato CSV;
- riutilizzare i file già scaricati per evitare download ridondanti.

Le tabelle gestite includono:

- market basket;
- anagrafica clienti;
- anagrafica prodotti;
- nomi prodotto;
- recommendation table del baseline esterno (ALS e BestCategory);
- tabelle per analisi esplorative.

---

### 2. Costruzione del grafo

La classe `MarketBasketGraphBuilder` trasforma i CSV in un oggetto `HeteroData` di PyTorch Geometric.

#### Operazioni principali

- caricamento dei file CSV;
- parsing temporale della colonna data;
- rimozione di duplicati;
- costruzione delle mappature:
  - `user_map`
  - `item_map`
  - mappe inverse;
- creazione degli archi user-item;
- aggiunta del timestamp sugli archi;
- costruzione delle feature di nodo per utenti e prodotti;
- salvataggio del grafo processato su disco.

#### Feature supportate

- feature categoriche codificate tramite factorization;
- feature numeriche convertite in `float`;
- cardinalità delle feature categoriche salvate per l’uso nel modello.

Il grafo finale viene convertito in **non orientato** tramite trasformazione `ToUndirected()`.

---

### 3. Schema condiviso del dataset

La classe `DataClass` centralizza i nomi standard usati in tutto il progetto:

- nome della colonna utente;
- nome della colonna prodotto;
- nome della colonna temporale;
- nomi delle relazioni del grafo;
- liste di feature categoriche e numeriche per utenti e prodotti.

Questo consente a tutti i moduli della pipeline di lavorare con una convenzione comune.

---

### 4. Creazione degli split temporali e dei loader

La classe `GraphLoader` gestisce il passaggio dal grafo completo ai dati pronti per training e valutazione.

#### Funzionalità principali

- ordinamento degli archi per tempo;
- split temporale in:
  - training;
  - validation;
  - test;
- costruzione dei sottografi di contesto;
- preservazione degli identificativi globali dei nodi (`n_id`);
- creazione di `LinkNeighborLoader` per il training e la validation;
- costruzione di:
  - `edge_label_index` per validation e test;
  - `exclude_links` per evitare leakage e raccomandazioni su archi già osservati.

#### Controllo leakage

La classe contiene una logica esplicita per:

- rimuovere dal contesto gli archi che appartengono al target da predire;
- verificare che non ci sia overlap tra ground truth e link esclusi;
- sollevare errore in caso di leakage.

Questa parte è centrale per garantire correttezza sperimentale.

---

## Rete neurale

### RecommenderModel

Il cuore del progetto è la classe `RecommenderModel`, una rete neurale per raccomandazioni su grafo eterogeneo. Tale modello è ottimizzato per la prototipizzazione, permettendo la selezione dei parametri, che vengono inseriti in MLflow.

#### Input supportati

Il modello può inizializzare i nodi in due modi:

1. **Embedding liberi**
   - un embedding per utente;
   - un embedding per prodotto.

2. **Embedding derivati da feature**
   - embedding per feature categoriche;
   - layer lineari per feature numeriche;
   - eventuali componenti stocastiche aggiuntive libere per utente e prodotto.

#### Backbone GNN

La parte convoluzionale usa uno stack di `HeteroConv` con `SAGEConv` nelle due direzioni:

- utente $$\rightarrow$$ prodotto;
- prodotto $$\rightarrow$$ utente.

#### Proiezione finale

Dopo la propagazione sul grafo, gli embedding passano attraverso layer di proiezione:

- condivisi, se `two_tower=False`;
- separati per utenti e prodotti, se `two_tower=True`.

Le proiezioni possono usare:

- `LayerNorm`;
- `BatchNorm`;
- nessuna normalizzazione.

#### Modalità di output

- **binary**: il decoder restituisce uno score user-item tramite prodotto scalare;
- **triplet**: restituisce direttamente embedding normalizzati.

---

## Loss functions supportate

La codebase supporta differenti funzioni obiettivo, selezionabili da configurazione.

### 1. `TripletMarginLoss`

Loss metrica classica su triplette anchor-positive-negative.

Formula:

$$
L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}
$$

dove:

$$
d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p
$$

Dove:

- $$a$$ è l’anchor;
- $$p$$ è il positivo;
- $$n$$ è il negativo;
- $$m$$ è il margine;
- $$p$$ nella distanza è il grado della norma.

L’obiettivo è imporre che il negativo sia più distante dell’item positivo di almeno $$m$$.

### 2. `BPRLoss`

Bayesian Personalized Ranking loss per ranking pairwise implicito con eventuale termine di regolarizzazione L2.

Per uno score positivo $$s^+$$ e uno negativo $$s^-$$:

$$
\mathcal{L}_{BPR} = -\log \sigma(s^+ - s^-)
$$

con:

$$
s^+ = a^T p, \qquad s^- = a^T n
$$

Se è attiva la regolarizzazione L2:

$$
\mathcal{L}_{reg} = \lambda \left( \lVert a \rVert_2^2 + \lVert p \rVert_2^2 + \lVert n \rVert_2^2 \right)
$$

La loss finale implementata è:

$$
\mathcal{L} = \mathcal{L}_{BPR} + \mathcal{L}_{reg}
$$

Questa loss è adatta a problemi di ranking implicito, dove interessa ordinare il positivo sopra il negativo più che stimare una probabilità calibrata.

### 3. `InfoNCE`

Loss contrastiva per apprendimento di embedding discriminativi.

Formula:

$$
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\log
\frac{
\exp\left(\mathrm{sim}(q, k^+)/\tau\right)
}{
\exp\left(\mathrm{sim}(q, k^+)/\tau\right)
+
\sum_{i=1}^{N}
\exp\left(\mathrm{sim}(q, k_i^-)/\tau\right)
}
$$

### 4. `FocalLoss`

Loss per classificazione sbilanciata:

$$
\mathrm{FL}(p_t) = -\alpha_t (1-p_t)^{\gamma}\log(p_t)
$$

dove:

- $$p_t$$ è la probabilità assegnata alla classe corretta;
- $$\alpha_t$$ è un fattore di bilanciamento tra classi;
- $$\gamma$$ è il parametro di focusing, che riduce il peso degli esempi facili.

Nel caso binario:

$$
p_t =
\begin{cases}
p & \text{se } y=1 \\
1-p & \text{se } y=0
\end{cases}
$$

e

$$
\alpha_t =
\begin{cases}
\alpha & \text{se } y=1 \\
1-\alpha & \text{se } y=0
\end{cases}
$$

### 5. `Binary Cross Entropy`

Binary Cross Entropy con logits.

Per un logit $$x$$ e target $$y \in \{0,1\}$$:

$$
\mathcal{L}_{BCE}(x,y) = - \left[ y \log \sigma(x) + (1-y) \log (1-\sigma(x)) \right]
$$

Dove:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

La versione `BCEWithLogitsLoss` combina sigmoid e cross-entropy in una forma numericamente più stabile.

### 6. `MultiLoss`

Permette la combinazione di due loss.

Formula:

$$
\mathcal{L}_{tot} = \mathcal{L}_{\text{Prima loss}} + \alpha \cdot \mathcal{L}_{\text{Seconda loss}}
$$

---

## Addestramento

La classe `Training` implementa le routine di training e validation.

### Regimi supportati

#### Binary training

- il modello restituisce logits sugli archi candidati;
- la loss viene calcolata rispetto alle label positive/negative.

#### Contrastive / triplet training

- il modello restituisce embedding;
- il batch viene trasformato in:
  - anchor;
  - positive;
  - negative;
- la loss viene calcolata su triplette.

### Caratteristiche della fase di training

- mixed precision tramite `autocast`;
- uso di `GradScaler`;
- optimizer `Adam`;
- scheduler `CosineAnnealingLR`;
- warmup lineare con `pytorch_warmup`.

---

## Inferenza

La classe `Inference` si occupa della generazione delle raccomandazioni top-k.

### Processo

1. encoding del sottografo tramite `model.encoder()`;
2. estrazione di embedding utente e prodotto;
3. costruzione della matrice score;
4. mascheramento dei link da escludere;
5. selezione top-k tramite `topk()`.

### Similarità

La logica attuale supporta ranking basato su prodotto scalare e, nel caso di similarità coseno, normalizza gli embedding prima del calcolo dello score.

Dato un insieme di embedding utente $$Q \in \mathbb{R}^{N_u \times d}$$ e un insieme di embedding prodotto $$D \in \mathbb{R}^{N_i \times d}$$, la matrice degli score viene calcolata come:

$$
S = QD^T
$$

Nel caso di similarità coseno, gli embedding vengono prima normalizzati:

$$
\hat{q}_u = \frac{q_u}{\lVert q_u \rVert_2}, \qquad \hat{d}_i = \frac{d_i}{\lVert d_i \rVert_2}
$$

per cui lo score diventa:

$$
s(u,i) = \hat{q}_u^T \hat{d}_i
$$

Gli archi già osservati o da escludere vengono mascherati imponendo:

$$
s(u,i) = -\infty
$$

così da impedirne la selezione nella top-k finale.

---

## Dettaglio matematico del modello

### Message passing e convoluzione GraphSAGE eterogenea

Il modello utilizza layer `HeteroConv` con operatori `SAGEConv` distinti per le due relazioni:

- $$\text{user} \rightarrow \text{item}$$
- $$\text{item} \rightarrow \text{user}$$

Per un nodo target $$v$$, GraphSAGE aggrega i vicini $$\mathcal{N}(v)$$ e combina l’informazione del nodo con quella del suo intorno. In forma semplificata:

$$
m_v^{(l)} = \operatorname{AGGR}\left( \left\{ h_u^{(l)} : u \in \mathcal{N}(v) \right\} \right)
$$

$$
h_v^{(l+1)} = \sigma \left( W_{self}^{(l)} h_v^{(l)} + W_{neigh}^{(l)} m_v^{(l)} \right)
$$

oppure, nella formulazione concatenata tipica di GraphSAGE:

$$
h_v^{(l+1)} = \sigma \left( W^{(l)} \cdot \left[ h_v^{(l)} \; \Vert \; m_v^{(l)} \right] \right)
$$

Dove:

- $$h_v^{(l)}$$ è la rappresentazione del nodo $$v$$ al layer $$l$$;
- $$\mathcal{N}(v)$$ è l’insieme dei vicini campionati;
- $$\operatorname{AGGR}$$ è una funzione di aggregazione, ad esempio `mean` o `max`;
- $$\sigma$$ è una non linearità;
- $$W_{self}^{(l)}$$ e $$W_{neigh}^{(l)}$$ sono parametri appresi.

Nel caso eterogeneo, il processo viene applicato separatamente per ciascun tipo di relazione e i contributi vengono poi combinati dal layer `HeteroConv`.

### Inizializzazione degli embedding

Il modello può partire da:

#### 1. Embedding liberi

Per utenti e prodotti vengono apprese matrici di embedding:

$$
E_u \in \mathbb{R}^{|U| \times d_{in}}, \qquad E_i \in \mathbb{R}^{|I| \times d_{in}}
$$

Dove:

- $$|U|$$ è il numero di utenti;
- $$|I|$$ è il numero di prodotti;
- $$d_{in}$$ è la dimensionalità iniziale.

#### 2. Feature-based embeddings

Se sono presenti feature categoriche e numeriche:

- una feature categorica $$x_c$$ viene trasformata tramite una embedding lookup;
- una feature numerica $$x_n$$ viene trasformata tramite un layer lineare.

Le componenti vengono poi concatenate:

$$
h_v^{(0)} = \left[ e_{c_1} \Vert e_{c_2} \Vert \dots \Vert W_1 x_{n_1} \Vert W_2 x_{n_2} \Vert \dots \right]
$$

Eventuali embedding stocastici aggiuntivi vengono concatenati alla rappresentazione iniziale.

### Layer di proiezione

Dopo i layer convoluzionali, le rappresentazioni vengono proiettate nello spazio latente finale tramite una MLP.

Con normalizzazione opzionale, uno strato tipico ha forma:

$$
z^{(l+1)} = \operatorname{Dropout} \left( \operatorname{ReLU} \left( \operatorname{Norm} \left( W^{(l)} z^{(l)} + b^{(l)} \right) \right) \right)
$$

L’ultimo layer proietta in $$d_{out}$$:

$$
z = W_{out} z^{(L)} + b_{out}
$$

In configurazione `two_tower=True`, utenti e prodotti hanno due teste distinte:

$$
z_u = f_u(h_u), \qquad z_i = f_i(h_i)
$$

altrimenti condividono la stessa testa di proiezione.

### Decoder per link prediction

In modalità `binary`, lo score tra utente $$u$$ e prodotto $$i$$ è il prodotto scalare tra embedding finali:

$$
s(u,i) = z_u^T z_i
$$

Tale score viene poi passato alla loss binaria, ad esempio `BCEWithLogitsLoss`.

In modalità `triplet`, il modello restituisce direttamente embedding normalizzati:

$$
\tilde{z}_u = \frac{z_u}{\lVert z_u \rVert_2}, \qquad \tilde{z}_i = \frac{z_i}{\lVert z_i \rVert_2}
$$

---

## Valutazione

La classe `Evaluation` calcola metriche ranking top-k a partire da:

- matrice delle predizioni $$\text{pred}\in\mathbb{N}^{N_u \times k}$$;
- ground truth sotto forma di `edge_label_index`.

### Metriche principali

#### Precision@K

$$
\operatorname{Precision@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|Rel_u \cap Rec_u^K|}{K}
$$

#### Recall@K

$$
\operatorname{Recall@K} = \frac{1}{|U|} \sum_{u \in U} \frac{|Rel_u \cap Rec_u^K|}{|Rel_u|}
$$

#### MAP@K

Media della precisione calcolata ai rank rilevanti fino a $$K$$.

#### Hit Ratio@K

$$
\operatorname{Hit@K} = \frac{1}{|U|} \sum_{u \in U} \mathbf{1}\left(|Rel_u \cap Rec_u^K| > 0\right)
$$

#### NDCG@K

Misura la qualità dell’ordinamento pesando maggiormente gli item rilevanti ai primi posti.

#### Average Popularity@K

Misura quanto le raccomandazioni tendono verso item popolari.

#### Personalization

Misura quanto le liste raccomandate per utenti diversi siano tra loro differenziate.

---

## Tracking esperimenti con MLflow

La classe `MlflowConfig` gestisce:

- tracking URI;
- nome esperimento;
- credenziali di accesso;
- caricamento di modelli salvati da uno specifico `run_id`;
- avvio condizionale di una run tramite context manager.

Il motore logga:

- hyperparameters;
- checkpoint di epoca;
- train loss;
- validation loss;
- metriche ranking di validation;
- metriche finali di test.

### Configurazione MLflow

Per utilizzare un’istanza locale di MLflow, è necessario definire le seguenti variabili d’ambiente:

- `MLFLOW_BACKEND_URI`: backend store dei metadati MLflow;
- `MLFLOW_ARTIFACT_ROOT`: directory dove salvare gli artifact;
- `MLFLOW_HOST`: host su cui esporre il server MLflow;
- `MLFLOW_PORT`: porta del server MLflow;
- `MLFLOW_TRACKING_URI`: URI usato dal client per il tracking;
- `MLFLOW_EXPERIMENT`: nome dell’esperimento.

Esempio di configurazione per istanza locale:

```env
MLFLOW_BACKEND_URI=sqlite:////path/mlflow.db
MLFLOW_ARTIFACT_ROOT=file:///path/artifacts
MLFLOW_HOST=127.0.0.1
MLFLOW_PORT=1234
MLFLOW_TRACKING_URI=http://127.0.0.1:1234
MLFLOW_EXPERIMENT=NomeEsperimento
```

Esempio di configurazione per istanza remota:

```env
MLFLOW_EXPERIMENT=progetto
MLFLOW_TRACKING_USERNAME=esperimento
MLFLOW_TRACKING_PASSWORD=password
MLFLOW_TRACKING_URI=https://istanza-mlflow
MLFLOW_TRACKING_INSECURE_TLS=true
```

### Avvio di un’istanza locale di MLflow

Se si desidera avviare MLflow in locale, è possibile esporre il server con una configurazione coerente con le variabili sopra definite. Ad esempio:

```bash
mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host "$MLFLOW_HOST" \
  --port "$MLFLOW_PORT"
```

In alternativa, il progetto può esporre un target dedicato nel `Makefile`, ad esempio `make mlflow`, per semplificare l’avvio del tracking server in ambiente locale.

---

## Configurazione

### HyperParams

La classe `HyperParams` carica i parametri da `interface.yaml`, unificando i blocchi:

- `training`
- `model`
- `graph`
- `loader`
- `eval`

Tra i parametri più importanti:

- learning rate;
- numero di epoche;
- margine della triplet loss;
- numero di layer convoluzionali;
- dimensioni di input, hidden e output;
- top-k di valutazione;
- numero di hop e vicini campionati;
- modalità (`binary` / `triplet`);
- tipo di similarità.

---

## Come utilizzare la rete

### 1. Preparare l’ambiente

Per l’ambito pacchetti:

- Python compatibile con il progetto;
- PyTorch installato;
- PyTorch Geometric installato.

In caso non dovessero essere installati, procedere con l’installazione via `uv` e `pyproject.toml`.

In seguito:

- accesso alle dipendenze necessarie (`mlflow`, `dotenv`, `yaml`, `pytorch_warmup`, ecc.);
- file di configurazione `interface.yaml` valorizzato;
- variabili ambiente MLflow e data path configurate.

### 2. Configurare `interface.yaml`

Il file deve contenere almeno:

- hyperparameters di training;
- hyperparameters del modello;
- parametri di costruzione del grafo;
- parametri del loader;
- parametri di evaluation;
- eventuale blocco `settings.mlflow` se si vuole caricare un modello da MLflow. In quel caso l’interfaccia deve essere allineata agli iperparametri sotto la sezione `graph`.

### 3. Scarico dei dati

La pipeline si aspetta di poter:

- scaricare i dataset tramite `DatasetManager`; oppure
- trovare i CSV già persistiti nelle cartelle previste.

In caso i dati non siano presenti, il `DatasetManager` procederà con lo scarico e la riassegnazione delle directory in base agli iperparametri.

### 4. Lanciare il training

Lo script principale espone due flag CLI:

- Esegue il training completo con logging MLflow attivo.

```bash
uv run python3 -m graph_recc_sys
```

- Esegue la pipeline senza logging su MLflow.

```bash
uv run python3 -m graph_recc_sys --dry-run
```

- Salta il training, carica un modello da MLflow ed esegue l’inferenza.

```bash
uv run python3 -m graph_recc_sys --no-training
```

### 5. Flusso di esecuzione runtime

A runtime avviene il seguente flusso:

1. caricamento hyperparameters;
2. configurazione MLflow;
3. download dataset e recommendation table;
4. costruzione del grafo eterogeneo;
5. creazione degli split temporali;
6. inizializzazione del modello;
7. training oppure caricamento del modello;
8. inferenza sul validation/test set;
9. calcolo metriche;
10. esportazione delle raccomandazioni finali in CSV.

### 6. Output principali

Durante o dopo l’esecuzione, il progetto produce:

- dataset grezzi persistiti in locale;
- grafo processato PyG;
- mapping utente/prodotto;
- artifact MLflow;
- metriche di validation e test;
- file CSV con raccomandazioni finali decodificate.

---

## Punto di ingresso della pipeline

Lo script principale:

- carica configurazioni e variabili ambiente;
- costruisce il dataset;
- costruisce il grafo;
- crea i loader;
- istanzia modello e engine;
- decide se eseguire training o inferenza;
- salva le raccomandazioni finali.

In altre parole, rappresenta il punto di orchestrazione operativo dell’intero progetto.

---

## Quando usare questa codebase

Questa codebase è adatta quando si vuole:

- modellare raccomandazioni come problema di **link prediction**;
- sfruttare informazione strutturale utente-item;
- includere feature di utenti e prodotti;
- eseguire split temporali rigorosi;
- confrontare qualità predittiva e proprietà comportamentali delle raccomandazioni;
- integrare tracking sperimentale con MLflow.

---

## Possibili estensioni future

Alcune direzioni naturali di evoluzione del progetto:

- supporto a più tipi di relazioni nel grafo;
- negative sampling più avanzato;
- ranking losses aggiuntive;
- calibrazione degli score;
- explainability delle raccomandazioni;
- serving online del modello.

---

## Conclusione

Il progetto fornisce una pipeline completa per recommendation su grafo eterogeneo, con forte attenzione a:

- correttezza dello split temporale;
- modularità della codebase;
- flessibilità del training;
- valutazione ranking top-k;
- tracciamento sperimentale.

La struttura a componenti separati rende il sistema estendibile, riusabile e adatto sia alla sperimentazione sia a un contesto operativo controllato.
