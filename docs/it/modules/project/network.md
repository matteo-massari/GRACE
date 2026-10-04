# Recommender Graph Neural Network

## Panoramica

Il modulo della **Recommender Graph Neural Network** implementa il cuore del sistema di raccomandazione basato su grafo eterogeneo user-item.

La rete neurale apprende rappresentazioni latenti di:

- utenti;
- prodotti;
- relazioni di acquisto tra utenti e prodotti.

A partire da tali embedding, il modello può essere usato sia per **link prediction binaria** sia per **apprendimento contrastivo / metrico**.

## Obiettivo

L’obiettivo della rete è modellare il comportamento di acquisto come problema di raccomandazione su grafo, sfruttando:

- struttura del grafo user-item;
- feature dei nodi, se disponibili;
- propagazione dell’informazione tramite convoluzioni Graph Neural Network;
- proiezione finale in uno spazio latente adatto al ranking o alla similarità.

## Architettura del modello

Il modello principale è la classe ``RecommenderModel``.

La pipeline interna del modello è composta da quattro blocchi:

1. costruzione delle rappresentazioni iniziali dei nodi;
1. propagazione tramite layer convoluzionali eterogenei;
1. proiezione nello spazio latente finale;
1. decodifica dello score user-item oppure restituzione degli embedding.

## Rappresentazioni iniziali dei nodi

Il modello supporta due strategie di inizializzazione.

### 1. Embedding liberi

Quando ``add_features=False``, utenti e prodotti vengono rappresentati tramite matrici di embedding apprese direttamente:

{math}`E_u \in \mathbb{R}^{|U| \times d_{in}}, \qquad E_i \in \mathbb{R}^{|I| \times d_{in}}`

Dove:

- {math}`|U|` è il numero di utenti;
- {math}`|I|` è il numero di prodotti;
- {math}`d_{in}` è la dimensionalità di input.

### 2. Embedding basati su feature

Quando ``add_features=True``, le feature categoriali vengono codificate tramite layer di embedding e le feature numeriche tramite layer lineari.

La rappresentazione iniziale di un nodo può essere espressa come concatenazione di componenti:

{math}`h_v^{(0)} = \left[ e_{c_1} \Vert e_{c_2} \Vert \dots \Vert W_1 x_{n_1} \Vert W_2 x_{n_2} \Vert \dots \right]`

Il modello può inoltre concatenare embedding stocastici aggiuntivi per utenti e prodotti, utili ad aumentare la capacità rappresentativa anche in presenza di poche feature informative.

## Convoluzione eterogenea

La rete usa uno stack di layer ``HeteroConv`` con operatori ``SAGEConv`` nelle due direzioni:

- ``user -> item``
- ``item -> user``

Per un nodo target {math}`v`, GraphSAGE aggrega l’informazione dai vicini {math}`\mathcal{N}(v)`:

{math}`m_v^{(l)} = \operatorname{AGGR}\left( \left\{ h_u^{(l)} : u \in \mathcal{N}(v) \right\} \right)`

Nell'implementazione, `SAGEConv` usa `project=True` e `normalize=True`.
Una forma schematica del calcolo per ciascuna relazione è:

```{math}
\hat h_u = \operatorname{ReLU}(W_p h_u + b_p), \qquad
m_v = \operatorname{AGGR}_{u \in \mathcal{N}(v)} \hat h_u
```

```{math}
h'_v = \operatorname{normalize}(W_r h_v + W_n m_v + b)
```

`HeteroConv` combina i contributi delle relazioni. Il codice applica dropout
dopo ogni convoluzione; non aggiunge un'ulteriore ReLU dopo `HeteroConv`.
L'aggregatore è controllato da `aggr`, con `mean` nella configurazione fornita.

## Layer di proiezione

Dopo la convoluzione, le rappresentazioni vengono trasformate tramite uno stack di layer fully connected.

Uno strato tipico segue la forma:

{math}`z^{(l+1)} = \operatorname{Dropout} \left( \operatorname{ReLU} \left( \operatorname{Norm} \left( W^{(l)} z^{(l)} + b^{(l)} \right) \right) \right)`

L’ultimo layer proietta nello spazio latente finale:

{math}`z = W_{out} z^{(L)} + b_{out}`

Il modello può operare in due configurazioni:

- **single tower**: utenti e prodotti condividono la stessa testa di proiezione;
- **two tower**: utenti e prodotti hanno due teste separate.

In configurazione two-tower:

{math}`z_u = f_u(h_u), \qquad z_i = f_i(h_i)`

## Decoder

### Modalità binary

In modalità ``binary``, lo score tra un utente {math}`u` e un prodotto {math}`i` è il prodotto scalare tra embedding finali:

{math}`s(u,i) = z_u^T z_i`

Questo score viene poi usato in una loss binaria per distinguere archi positivi e negativi.

### Modalità triplet

In modalità ``triplet``, il modello restituisce direttamente embedding normalizzati:

{math}`\tilde{z}_u = \frac{z_u}{\lVert z_u \rVert_2}, \qquad \tilde{z}_i = \frac{z_i}{\lVert z_i \rVert_2}`

Questi embedding possono essere usati con loss metriche o contrastive.

## Loss supportate

La loss è selezionata da `loss_function`. L'entry point imposta `binary` per
`BCEWithLogitsLoss` e `FocalLoss`, e `triplet` per le altre loss.
Le formule seguenti ne riassumono l'obiettivo; riduzioni e regolarizzazione
seguono le classi usate nel codice. Nel caso BPR l'engine usa `reg_weight=0.1`;
`MultiLoss` usa i propri valori predefiniti, inclusi `alpha=0.5` e margine 4.0.

### Triplet Margin Loss

{math}`\mathcal{L}_{triplet}(a,p,n) = \max \left( d(a,p) - d(a,n) + m, 0 \right)`

Dove {math}`a` è l’anchor, {math}`p` il positivo, {math}`n` il negativo e {math}`m` il margine.

### BPR Loss

{math}`\mathcal{L}_{BPR} = -\log \sigma(s^+ - s^-)`

con:

{math}`s^+ = a^T p, \qquad s^- = a^T n`

### InfoNCE

{math}`\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\operatorname{sim}(a,p)/\tau)}{\exp(\operatorname{sim}(a,p)/\tau) + \sum_{n \in \mathcal{N}} \exp(\operatorname{sim}(a,n)/\tau)}`

### BCE with logits

{math}`\mathcal{L}_{BCE}(x,y) = - \left[ y \log \sigma(x) + (1-y) \log (1-\sigma(x)) \right]`

### Focal Loss

{math}`\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)`

### MultiLoss

{math}`\mathcal{L}_{tot} = \mathcal{L}_{BPR} + \alpha \cdot \mathcal{L}_{triplet}`

## Dati e split temporale

`DatasetManager` legge i CSV locali. `MarketBasketGraphBuilder` ordina gli acquisti
per data, conserva il primo evento di ogni coppia cliente–prodotto e costruisce
le due relazioni `buys` e `rev_buys`. Il grafo PyG è distinto da Neo4j.

I nodi sono `ids_customer` e `ids_product`. Nella configurazione corrente le
feature cliente sono `sector`, `potential`, `county`, `creditlimit`, trattate
come categorie; i prodotti usano un embedding appreso. Gli ID cliente non sono
una feature categoriale del percorso `add_features=True`. Mapping e grafo
processato vengono salvati localmente; la cache viene ricostruita se cambiano i
CSV sorgente. Modifiche al preprocessing richiedono una ricostruzione esplicita
della cache.

`GraphLoader` ordina gli archi per timestamp e usa il 60% iniziale per training,
il 20% successivo per validation e il 20% finale per test, salvo arrotondamenti.
Il taglio è per numero di archi, non per date di calendario: eventi con lo stesso
timestamp possono ricadere in split diversi. Validation e test usano entrambi
il contesto di training, senza aggiungere gli acquisti della validation al test.

| Sampler | Comportamento |
| --- | --- |
| `neighbor` | `LinkNeighborLoader` temporale con cutoff precedente al target; richiede `pyg-lib` compatibile con PyTorch |
| `full` | Tutti i nodi, target in ordine cronologico e solo archi precedenti al primo target del batch; pensato per piccoli dataset |

Il sampler completo è conservativo sul contesto e non riproduce il campionamento
dei vicini. I negativi escludono i positivi del contesto e del batch; ciò non
significa che siano prodotti che il cliente non acquisterà mai in futuro.

## Inferenza e ranking

L'inferenza usa `model.encoder()` e calcola gli score per tutte le coppie
cliente–prodotto. `similarity` seleziona una delle tre funzioni:

```{math}
s(u,i) =
\begin{cases}
-\lVert z_u-z_i\rVert_2 & \text{euclidean} \\
\dfrac{z_u^Tz_i}{\lVert z_u\rVert_2\lVert z_i\rVert_2} & \text{cosine} \\
z_u^Tz_i & \text{dot}
\end{cases}
```

Il percorso `euclidean` usa realmente la distanza euclidea negativa, non il
prodotto scalare. Il ranking lavora sugli embedding dell'encoder; la
normalizzazione del `forward` triplet è un passaggio distinto usato nel training.

Gli archi esclusi ricevono score meno infinito prima della top-k. Con il grafo
preprocessato, i prodotti già comprati nel contesto di training non sono
candidati. Se un cliente ha meno di `k` prodotti disponibili, l'inferenza segnala
un errore anziché restituire prodotti esclusi. Il calcolo della matrice completa
degli score va considerato nel dimensionamento di dataset grandi.

## Valutazione, tracking ed export

| Metrica | Significato |
| --- | --- |
| Precision@K | Quota delle raccomandazioni presente nei target |
| Recall@K | Quota dei target recuperata nelle raccomandazioni |
| MAP@K | Qualità dell'ordinamento dei risultati rilevanti |
| Hit ratio@K | Presenza di almeno un risultato rilevante |
| NDCG@K | Rilevanza con sconto per posizione e normalizzazione |
| Average popularity | Popolarità media dei prodotti raccomandati nel grafo di training |
| Personalization | Diversità fra le liste di raccomandazioni degli utenti |

L'implementazione delle metriche è quella delle classi `LinkPred*` di PyTorch
Geometric. L'engine usa Adam, scheduler cosine e warmup lineare. Ogni epoca
calcola loss e metriche di validation; alla fine calcola quelle del test.

Con tracking attivo, MLflow riceve iperparametri, checkpoint per epoca e
`final_model`. `--dry-run` disattiva solo il tracking. La selezione automatica
del dispositivo usa CUDA, poi MPS, infine CPU; `--device` permette
la scelta esplicita.

L'export CSV avviene nel percorso `--no-training`, dopo il caricamento del
modello da MLflow. Contiene `ids_customer`, `ids_product` e `product_name`, nel
percorso `DATA_PATH/rec_tables/<k>/dataset_res_<k>_features_<add_features>.csv`.
Non contiene score o rank espliciti. Il contesto è quello del test: non è un
riaddestramento su tutto lo storico. Per collegare l'output al Knowledge Graph
si esegue poi `make pipeline-recommendations`.

## Integrazione nella pipeline

La rete neurale non opera in isolamento, ma si inserisce nella pipeline completa del progetto:

1. il grafo viene costruito a partire da market basket e anagrafiche;
1. il loader crea split temporali e mini-batch;
1. il modello apprende embedding user-item;
1. l’inferenza genera raccomandazioni top-k;
1. il modulo di evaluation calcola metriche ranking.

## Parametri principali

Tra gli hyperparameters più rilevanti del modello:

- ``input_channels``: dimensione iniziale degli embedding;
- ``hidden_channels``: dimensione dei layer convoluzionali;
- ``out_channels``: dimensione finale dello spazio latente;
- ``n_conv_layer``: numero di layer GraphSAGE eterogenei;
- ``n_projs_layer``: numero di layer di proiezione;
- ``dropout_rate``: dropout applicato dopo convoluzioni e proiezioni;
- ``two_tower``: uso di teste separate per utenti e prodotti;
- ``proj_norm``: normalizzazione nei layer di proiezione;
- ``aggr``: funzione di aggregazione nei layer SAGEConv;
- ``add_features``: uso o meno delle feature di nodo.

## Quando usare questa rete

Questa architettura è adatta quando si vuole:

- trattare la recommendation come problema di **link prediction**;
- sfruttare contemporaneamente grafo e feature tabellari;
- usare split temporali rigorosi;
- confrontare training binario e training contrastivo;
- produrre embedding latenti riutilizzabili anche in downstream task.

```{note}
La qualità finale del sistema dipende non solo dall’architettura della rete, ma anche dalla correttezza dello split temporale, dalla qualità delle feature e dalla strategia di negative sampling adottata durante training e valutazione.
```
