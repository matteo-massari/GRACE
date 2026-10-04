# Knowledge Graph e pipeline

Il Knowledge Graph è il database Neo4j consultato da Graph RAG. Rappresenta
clienti, prodotti `Level5`, gerarchie di prodotto, rete vendita e geografia.
È distinto dal grafo cliente–prodotto PyTorch Geometric della GNN: nella demo
entrambi sono costruiti a partire dagli stessi CSV sintetici.

## Schema implementato

| Entità | Identificativo principale | Ruolo |
| --- | --- | --- |
| `Customer` | `ids_customer` | Cliente e attributi commerciali |
| `Level5` | `level5id` | Prodotto/categoria raccomandabile |
| `Level1`–`Level4` | `level1id`–`level4id` | Gerarchia dei prodotti |
| `SalesRep` | `ids_salesrep` | Venditore e assegnazioni clienti |
| `District`, `Territory`, `MacroRegion` | ID organizzativi | Struttura commerciale |
| `City`, `County`, `Country` | Codici/nome geografici | Localizzazione |
| `Sector`, `Branch`, `Channel`, `MarketSegment`, `Job` | Attributi dimensionali | Classificazione commerciale |

Il loader dimostrativo non crea una distinta entità `Material`: i prodotti sono
rappresentati come `Level5`. Le relazioni principali sono illustrate qui sotto;
sono presenti anche collegamenti `Channel`–`Sector` e `Sector`–`MarketSegment`.

```{mermaid}

graph LR

    %% Prodotti
    subgraph Prodotti
        Level5 -->|BELONGS_TO| Level4
        Level4 -->|BELONGS_TO| Level3
        Level3 -->|BELONGS_TO| Level2
        Level2 -->|BELONGS_TO| Level1
    end

    %% Clienti
    subgraph Clienti
        Customer -->|IN_SECTOR| Sector
        Customer -->|IN_BRANCH| Branch
        Customer -->|IN_CHANNEL| Channel
        Customer -->|IN_MARKET_SEGMENT| MarketSegment
        Customer -->|LOCATED_IN| City
    end

    %% Venditori
    subgraph Venditori
        SalesRep -->|TYPE| Job
        SalesRep -->|IN_CITY| City
        SalesRep -->|IN_CHANNEL| Channel
        SalesRep -->|IN_SECTOR| Sector
        SalesRep -->|IN_MARKET_SEGMENT| MarketSegment
        SalesRep -->|IN_DISTRICT| District
        District -->|IN_TERRITORY| Territory
        Territory -->|IN_MACRO_REGION| MacroRegion
        SalesRep -->|HAS_CUSTOMER| Customer
    end

    %% Organizzazione vendite
    subgraph Organizzazione
        MacroRegion -->|MANAGED_BY| SalesRep
        Territory -->|MANAGED_BY| SalesRep
        District -->|MANAGED_BY| SalesRep
    end

    %% Geografia
    subgraph Geografia
        City -->|IN_COUNTY| County
        County -->|IN_COUNTRY| Country
    end

    %% Relazioni prodotti
    Level5 -->|RECOMMENDED_TO| Customer
    Customer -->|BUYS| Level5

    %% Stili
    classDef product fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1;
    classDef customer fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20;
    classDef sales fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100;
    classDef salesHierarchy fill:#FBE9E7,stroke:#D84315,stroke-width:2px,color:#BF360C;
    classDef geo fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C;
    classDef shared fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#263238;

    class Level1,Level2,Level3,Level4,Level5 product;
    class Customer,Branch customer;
    class SalesRep,Job sales;
    class District,Territory,MacroRegion salesHierarchy;
    class City,County,Country geo;
    class Sector,Channel,MarketSegment shared;
```

`Customer -[:BUYS]-> Level5` descrive gli acquisti.
`Level5 -[:RECOMMENDED_TO]-> Customer` descrive le raccomandazioni importate.
L'assenza di una relazione indica che quell'evidenza non è disponibile nel grafo;
non dimostra che un acquisto sia impossibile o che il prodotto sia inadatto.

## Pipeline dei dati sintetici

1. `generate_synthetic_database.py` genera 100 clienti, 60 prodotti, 30 venditori
   e 5.453 eventi di acquisto con un seed fisso.
2. `load_synthetic_neo4j.py` legge i CSV, crea vincoli di unicità ed esegue
   transazioni a batch con `MERGE` per nodi e relazioni.
3. Gli acquisti ripetuti della stessa coppia vengono aggregati nella relazione
   `BUYS`, conservando prima/ultima data, numero di eventi e quantità totale.
4. Il loader collega anagrafiche, gerarchie e venditori ai clienti.

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
```

Il caricamento è progettato per poter essere ripetuto senza duplicare le stesse
entità e relazioni. È un aggiornamento per inserimento/modifica: non elimina
automaticamente elementi scomparsi dai CSV. L'idempotenza completa sul database
non è stata verificata dal vivo; vedere il [resoconto](../../verification.md).

## Raccomandazioni della rete e placeholder

Dopo l'inferenza GNN, `make pipeline-recommendations` avvia `ETL_recc.py`.
L'importatore cerca clienti e prodotti esistenti e crea/aggiorna
`RECOMMENDED_TO` con il nome del prodotto. La destinazione dei prodotti è
`Level5.level5id`, corrispondente a `ids_product` nel CSV della rete.

Il percorso predefinito è
`database/gnn/rec_tables/10/dataset_res_10_features_True.csv`.
Per un'altra configurazione di output, impostare `RECOMMENDATIONS_CSV_PATH`.
Gli ID devono corrispondere alle anagrafiche già caricate.

Il generatore produce separatamente `recommendations_placeholder.csv`, marcato
`source=placeholder_not_gnn`. Non è un risultato del modello e non è importato
con il comando standard. Per una prova esplicita del grafo:

```bash
uv run python pipeline/load_synthetic_neo4j.py --include-placeholder-recommendations
```

L'importatore GNN non cancella le raccomandazioni precedenti e non azzera gli
attributi dei placeholder. Una nuova importazione non sostituisce quindi una
snapshot completa della top-k: mescolare placeholder e risultati reali nello
stesso database richiede una gestione esplicita delle relazioni precedenti.

## Neo4j e sorgenti private

`docker-compose.neo4j.yml` espone Neo4j e APOC Core con dati persistenti.
`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` e `NEO4J_DATABASE` configurano i
client. Le porte e l'avvio sono descritti nella [guida operativa](../getting_started.md).

`ETL_fact.py` conserva il template di estrazione aziendale, con SQL e driver
Impala da completare. Segnala questa limitazione prima di avviare connessioni.
Anche il modulo privato `graph_recc_sys.query` e `maisutils` non sono inclusi:
non servono per il percorso dimostrativo basato su CSV.
