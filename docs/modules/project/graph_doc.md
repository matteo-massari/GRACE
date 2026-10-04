# Knowledge Graph and pipeline

Knowledge Graph is the Neo4j database consulted by Graph RAG. Representation
customers, products `Level5`, product hierarchy, sales network and geography.
It is distinguished by the customer graph–product PyTorch Geometric of GNN: in the demo
both are built from the same synthetic CSVs.

## Scheme implemented

| Entity | Main identifier | Role |
| --- | --- | --- |
| `Customer` | `ids_customer` | Customer and commercial attributes |
| `Level5` | `level5id` | Recommended product/category |
| `Level1`–`Level4` | `level1id`–`level4id` | Hierarchy of products |
| `SalesRep` | `ids_salesrep` | Seller and customer assignments |
| `District`, `Territory`, `MacroRegion` | Organizational identifiers | Commercial structure |
| `City`, `County`, `Country` | Geographical codes | Location |
| `Sector`, `Branch`, `Channel`, `MarketSegment`, `Job` | Dimension attributes | Commercial classification |

The demonstrative loader does not create a distinct entity `Material`: products are
represented `Level5`. The main reports are illustrated below;
There are also connections `Channel`–`Sector` and `Sector`–`MarketSegment`.

```{mermaid}

graph LR

    %% Products
    subgraph Products
        Level5 -->|BELONGS_TO| Level4
        Level4 -->|BELONGS_TO| Level3
        Level3 -->|BELONGS_TO| Level2
        Level2 -->|BELONGS_TO| Level1
    end

    %% Customers
    subgraph Customers
        Customer -->|IN_SECTOR| Sector
        Customer -->|IN_BRANCH| Branch
        Customer -->|IN_CHANNEL| Channel
        Customer -->|IN_MARKET_SEGMENT| MarketSegment
        Customer -->|LOCATED_IN| City
    end

    %% Sales representatives
    subgraph Sales representatives
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

    %% Sales organization
    subgraph Organization
        MacroRegion -->|MANAGED_BY| SalesRep
        Territory -->|MANAGED_BY| SalesRep
        District -->|MANAGED_BY| SalesRep
    end

    %% Geography
    subgraph Geography
        City -->|IN_COUNTY| County
        County -->|IN_COUNTRY| Country
    end

    %% Product relationships
    Level5 -->|RECOMMENDED_TO| Customer
    Customer -->|BUYS| Level5

    %% Styles
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

`Customer -[:BUYS]-> Level5` describes the purchases.
`Level5 -[:RECOMMENDED_TO]-> Customer` describes the imported recommendations.
The absence of a report indicates that the evidence is not available in the graph;
does not show that a purchase is impossible or that the product is unsuitable.

## Synthetic data pipeline

1. `generate_synthetic_database.py` generates 100 customers, 60 products, 30 sellers
   and 5,453 buying events with a fixed seed.
2. `load_synthetic_neo4j.py` reads CSVs, creates uniqueness constraints and performs
   Batch transactions with `MERGE` for knots and relationships.
3. Repeated purchases of the same pair are aggregated in the report
   `BUYS`, keeping before/last date, number of events and total quantity.
4. The loader connects graphics, hierarchies and sellers to customers.

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
```

Loading is designed to be repeated without duplicating the same
entities and relationships. It is an update for insertion/modification: does not delete
automatically items disappeared from CSV. The full application on the database
was not verified live; see the [report](../../verification.md).

## Network and placeholder recommendations

After the GNN inference, `make pipeline-recommendations` start `ETL_recc.py`.
The importer seeks existing customers and products and creates/update
`RECOMMENDED_TO` with the product name. The destination of products is
`Level5.level5id`, corresponding to `ids_product` in the network CSV.

The default path is
`database/gnn/rec_tables/10/dataset_res_10_features_True.csv`.
For another output configuration, set `RECOMMENDATIONS_CSV_PATH`.
IDs must match the already uploaded graphics.

The generator produces separately `recommendations_placeholder.csv`, marked
`source=placeholder_not_gnn`. It is not a result of the model and it is not imported
with the standard command. For an explicit proof of the graph:

```bash
uv run python pipeline/load_synthetic_neo4j.py --include-placeholder-recommendations
```

The GNN importer does not cancel the previous recommendations and does not remove the
placeholder attributes. A new import does not replace a
top-k snapshot: Mix placeholders and real results in
same database requires explicit management of previous reports.

## Neo4j and private sources

`docker-compose.neo4j.yml` exposes Neo4j and APOC Core with persistent data.
`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` and `NEO4J_DATABASE` configure the
client. Doors and startup are described in [operational guide](../getting_started.md).

`ETL_fact.py` preserves the corporate extraction template, with SQL and driver
Impala to be completed. Report this limitation before you start connections.
Also the private module `graph_recc_sys.query` and `maisutils` are not included:
do not serve for the CSV-based demonstration path.
