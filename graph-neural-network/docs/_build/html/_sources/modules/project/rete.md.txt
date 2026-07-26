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

#. costruzione delle rappresentazioni iniziali dei nodi;
#. propagazione tramite layer convoluzionali eterogenei;
#. proiezione nello spazio latente finale;
#. decodifica dello score user-item oppure restituzione degli embedding.

## Rappresentazioni iniziali dei nodi

Il modello supporta due strategie di inizializzazione.

### 1. Embedding liberi

Quando ``add_features=False``, utenti e prodotti vengono rappresentati tramite matrici di embedding apprese direttamente:

```{math}
E_u \in \mathbb{R}^{|U| \times d_{in}}, \qquad E_i \in \mathbb{R}^{|I| \times d_{in}}
```

Dove:

- {math}`|U|` è il numero di utenti;
- {math}`|I|` è il numero di prodotti;
- {math}`d_{in}` è la dimensionalità di input.

### 2. Embedding basati su feature

Quando ``add_features=True``, le feature categoriali vengono codificate tramite layer di embedding e le feature numeriche tramite layer lineari.

La rappresentazione iniziale di un nodo può essere espressa come concatenazione di componenti:

```{math}
h_v^{(0)} = \left[ e_{c_1} \Vert e_{c_2} \Vert \dots \Vert W_1 x_{n_1} \Vert W_2 x_{n_2} \Vert \dots \right]
```

Il modello può inoltre concatenare embedding stocastici aggiuntivi per utenti e prodotti, utili ad aumentare la capacità rappresentativa anche in presenza di poche feature informative.

## Convoluzione eterogenea

La rete usa uno stack di layer ``HeteroConv`` con operatori ``SAGEConv`` nelle due direzioni:

- ``user -> item``
- ``item -> user``

Per un nodo target ```{math} v ```, GraphSAGE aggrega l’informazione dai vicini ```{math} \mathcal{N}(v) ```:

```{math}
m_v^{(l)} = \operatorname{AGGR}\left( \left\{ h_u^{(l)} : u \in \mathcal{N}(v) \right\} \right)
```

La nuova rappresentazione del nodo viene poi aggiornata come:

```{math}
h_v^{(l+1)} = \sigma \left( W^{(l)} \cdot \left[ h_v^{(l)} \Vert m_v^{(l)} \right] \right)
```

Dove:

- {math}`h_v^{(l)}` è l’embedding del nodo al layer {math}`l `;
- {math}`m_v^{(l)}` è il messaggio aggregato dai vicini;
- {math}`\operatorname{AGGR}` può essere ad esempio ``mean`` o ``max``;
- {math}`\sigma` è una non linearità;
- {math}`W^{(l)}` è una matrice di pesi appresa.

Nel caso eterogeneo, ogni relazione ha il proprio operatore di convoluzione, e il layer ``HeteroConv`` combina i contributi provenienti dai diversi tipi di archi.

## Layer di proiezione

Dopo la convoluzione, le rappresentazioni vengono trasformate tramite uno stack di layer fully connected.

Uno strato tipico segue la forma:

```{math}
z^{(l+1)} = \operatorname{Dropout} \left( \operatorname{ReLU} \left( \operatorname{Norm} \left( W^{(l)} z^{(l)} + b^{(l)} \right) \right) \right)
```

L’ultimo layer proietta nello spazio latente finale:

```{math}
z = W_{out} z^{(L)} + b_{out}
```

Il modello può operare in due configurazioni:

- **single tower**: utenti e prodotti condividono la stessa testa di proiezione;
- **two tower**: utenti e prodotti hanno due teste separate.

In configurazione two-tower:

```{math}
z_u = f_u(h_u), \qquad z_i = f_i(h_i)
```

## Decoder

### Modalità binary

In modalità ``binary``, lo score tra un utente ```{math} u ``` e un prodotto ```{math} i ``` è il prodotto scalare tra embedding finali:

```{math}
s(u,i) = z_u^T z_i
```

Questo score viene poi usato in una loss binaria per distinguere archi positivi e negativi.

### Modalità triplet

In modalità ``triplet``, il modello restituisce direttamente embedding normalizzati:

```{math}
\tilde{z}_u = \frac{z_u}{\lVert z_u \rVert_2}, \qquad \tilde{z}_i = \frac{z_i}{\lVert z_i \rVert_2}
```

Questi embedding possono essere usati con loss metriche o contrastive.

## Loss supportate

La rete può essere addestrata con differenti funzioni obiettivo, selezionate via configurazione.

### Triplet Margin Loss

```{math}
\mathcal{L}_{triplet}(a,p,n) = \max \left( d(a,p) - d(a,n) + m, 0 \right)
```

Dove ```{math} a ``` è l’anchor, ```{math} p ``` il positivo, ```{math} n ``` il negativo e ```{math} m ``` il margine.

### BPR Loss

```{math}
\mathcal{L}_{BPR} = -\log \sigma(s^+ - s^-)
```

con:

```{math}
s^+ = a^T p, \qquad s^- = a^T n
```

### InfoNCE

```{math}
\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\operatorname{sim}(a,p)/\tau)}{\exp(\operatorname{sim}(a,p)/\tau) + \sum_{n \in \mathcal{N}} \exp(\operatorname{sim}(a,n)/\tau)}
```

### BCE with logits

```{math}
\mathcal{L}_{BCE}(x,y) = - \left[ y \log \sigma(x) + (1-y) \log (1-\sigma(x)) \right]
```

### Focal Loss

```{math}
\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)
```

### MultiLoss

```{math}
\mathcal{L}_{tot} = \mathcal{L}_{BPR} + \alpha \cdot \mathcal{L}_{triplet}
```

## Integrazione nella pipeline

La rete neurale non opera in isolamento, ma si inserisce nella pipeline completa del progetto:

#. il grafo viene costruito a partire da market basket e anagrafiche;
#. il loader crea split temporali e mini-batch;
#. il modello apprende embedding user-item;
#. l’inferenza genera raccomandazioni top-k;
#. il modulo di evaluation calcola metriche ranking.

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
