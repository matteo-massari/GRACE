# Dataset sintetico locale
[English version](README.md)

Questa cartella contiene dati interamente fittizi e deterministici, generati da
`pipeline/generate_synthetic_database.py`. Le colonne dei CSV principali sono
coerenti con le trasformazioni di `pipeline/ETL_fact.py` e `pipeline/ETL_recc.py`.

| File | Contenuto |
| --- | --- |
| `customers.csv` | 100 consumatori e attributi anagrafici/dimensionali. |
| `salesreps.csv` | 30 venditori, distretti, territori e macroregioni. |
| `products.csv` | 60 prodotti nella gerarchia `Level1`-`Level5`. |
| `customer_salesreps.csv` | Assegnazione coerente di ogni cliente a un venditore. |
| `purchases.csv` | Acquisti a livello di evento (10--100 per consumatore). |
| `recommendations_placeholder.csv` | Top-10 fittizia: non è generata dalla GNN e ha `source=placeholder_not_gnn`. |

`gnn/` contiene la stessa base nel formato atteso dal motore GNN. Per usarla,
impostare `DATA_PATH=database/gnn`; la GNN troverà il dataset
`dataset_None_4_12`. Gli output GNN sono separati dai placeholder e non vengono
sovrascritti dalla rigenerazione. I CSV generati e le cache sono esclusi da Git.

Rigenerazione e caricamento Neo4j:

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
```

Il loader usa `MERGE` per aggiornare le stesse entità senza duplicarle; non
elimina automaticamente dati scomparsi dai CSV. Per importare anche le relazioni placeholder
`RECOMMENDED_TO` (normalmente da lasciare alla GNN), usare:

```bash
uv run python pipeline/load_synthetic_neo4j.py --include-placeholder-recommendations
```

## Formato GNN

`gnn/training_tables/dataset_None_4_12/` contiene:

| File | Colonne principali |
| --- | --- |
| `basket_data.csv` | `ids_customer`, `ids_product`, `dat` |
| `customer.csv` | `ids_customer`, `sector`, `potential`, `county`, `creditlimit` |
| `product.csv` | `ids_product` |
| `product_names.csv` | `ids_product`, `product_name` |

La GNN conserva il primo acquisto di ogni coppia. Neo4j aggrega invece gli eventi
nella relazione `BUYS`, conservando conteggio, quantità e prima/ultima data.
Le due rappresentazioni partono dalla stessa base e servono obiettivi diversi.

Il CSV GNN compare solo dopo l'inferenza, sotto `gnn/rec_tables/`. Rigenerare i
dati non sovrascrive quell'output. Per il flusso completo vedere la
[guida operativa](../docs/it/modules/getting_started.md).
