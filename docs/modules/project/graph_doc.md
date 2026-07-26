# Knowledge Graph 

Il Knowledge Graph aziendale rappresenta il livello di conoscenza centrale dell’architettura sviluppata nel progetto. Il suo scopo è integrare informazioni provenienti da diverse sorgenti aziendali e rappresentarle attraverso entità e relazioni esplicite.
A differenza del grafo utilizzato dalla Graph Neural Network, costruito principalmente per apprendere i pattern di acquisto, il Knowledge Graph mantiene una visione più completa del dominio aziendale. Al suo interno vengono rappresentati clienti, prodotti, materiali, gerarchie commerciali, venditori e informazioni organizzative.
Il grafo viene utilizzato soprattutto dall’agente AI, che può recuperare informazioni contestuali e utilizzarle per interpretare e spiegare le raccomandazioni generate dal modello.


Il Knowledge Graph è stato progettato per:
- centralizzare dati provenienti da sorgenti eterogenee;
- rappresentare esplicitamente le relazioni tra le entità aziendali;
- rendere più semplici le interrogazioni che coinvolgono più livelli informativi;
- fornire all’agente AI una base di conoscenza strutturata;
- supportare la generazione di spiegazioni contestualizzate;
- separare la conoscenza aziendale dalla componente predittiva del recommender system.
Il Knowledge Graph non sostituisce quindi il modello di raccomandazione, ma lo completa. La GNN individua i prodotti potenzialmente rilevanti, mentre il grafo fornisce le informazioni necessarie per comprenderne il contesto.


Il Knowledge Graph è implementato in Neo4j seguendo il paradigma del Property Graph.
Il modello è composto da:
- nodi, che rappresentano le entità aziendali;
- relazioni, che descrivono i collegamenti tra le entità;
- etichette, che identificano il tipo di nodo;
- proprietà, che contengono gli attributi descrittivi.

Le principali entità rappresentate comprendono:
- clienti;
- materiali;
- prodotti;
- categorie della gerarchia commerciale;
- venditori;
- settori aziendali;
- aree geografiche;
Le relazioni permettono, per esempio, di rappresentare che un cliente acquista un materiale, che un materiale appartiene a una determinata categoria di prodotto oppure che un cliente è seguito da uno specifico venditore.

Ontologia aziendale
La struttura del Knowledge Graph è definita attraverso un’ontologia di dominio, che stabilisce quali entità possono essere rappresentate e quali relazioni possono esistere tra esse.
L’ontologia costituisce quindi lo schema concettuale del grafo e permette di mantenere una rappresentazione coerente della conoscenza aziendale.
Una versione semplificata dell’ontologia può essere rappresentata nel seguente modo:

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

L’ontologia definisce inoltre le proprietà principali associate alle diverse entità.
Un cliente può essere descritto attraverso attributi come:
- identificativo;
- settore;
- area geografica;
- potenziale commerciale;
- limite di credito.

Un materiale può invece includere:
- identificativo;
- descrizione;
- categoria di appartenenza;
- posizione nella gerarchia di prodotto.

Nel progetto viene adottata una rappresentazione orientata al closed-world assumption: il grafo viene considerato una rappresentazione controllata della conoscenza disponibile nei sistemi aziendali. L’assenza di una relazione non viene interpretata come conoscenza incompleta da inferire automaticamente, ma come informazione non presente nel dominio modellato.
Questa scelta rende il Knowledge Graph più semplice da controllare e più adatto a un contesto aziendale, nel quale le informazioni devono derivare da sorgenti verificate.



La costruzione del grafo segue una pipeline composta da quattro fasi principali:
Estrazione, durante la quale i dati vengono acquisiti dai sistemi aziendali.
Trasformazione, che comprende pulizia, normalizzazione, deduplicazione e anonimizzazione.
Modellazione, nella quale le informazioni vengono convertite in nodi e relazioni secondo l’ontologia.
Caricamento, attraverso il quale il grafo viene aggiornato all’interno di Neo4j.
Le principali sorgenti comprendono:
- anagrafiche clienti;
- catalogo prodotti;
- gerarchie commerciali;
- storico degli acquisti;
- rete vendita;
- informazioni geografiche e organizzative.
L’utilizzo di identificativi univoci consente di evitare la duplicazione delle entità e di mantenere coerenti gli aggiornamenti nel tempo.


L’architettura distingue chiaramente il Knowledge Graph dal grafo utilizzato per il recommender system.
Il Knowledge Graph contiene una rappresentazione estesa del dominio aziendale e viene utilizzato principalmente dall’agente AI.
Il Recommendation Graph è invece una vista più compatta, composta principalmente da:
- clienti;
- prodotti di Level 5;
- relazioni di acquisto.
Questa separazione permette alla GNN di lavorare su una struttura ottimizzata per il training, senza introdurre entità e relazioni non direttamente necessarie alla raccomandazione.

