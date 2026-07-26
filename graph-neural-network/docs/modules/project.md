# Architettura del Progetto

```{toctree}
:maxdepth: 1

Knowledge Graph <project/graph_doc>
Architettura della rete <project/rete>
Sales Agent <project/agent_doc>

```


Il progetto è costituito da tre componenti principali che collaborano tra loro per fornire un sistema di raccomandazione intelligente e spiegabile:



L'obiettivo è combinare la capacità di apprendere dai dati della GNN con la conoscenza strutturata presente nel grafo e le capacità di ragionamento di un agente basato su Large Language Model.


**Neo4j**
Neo4j rappresenta il punto centrale della conoscenza aziendale.
Al suo interno vengono memorizzate le informazioni relative a clienti, prodotti e relazioni tra le diverse entità, creando un grafo che descrive il dominio aziendale in maniera naturale.

Questa componente permette di:
- centralizzare la conoscenza;
- rappresentare relazioni complesse tra le entità;
- fornire il contesto necessario sia alla GNN sia all'agente AI.

Neo4j non è solamente un database, ma costituisce la base di conoscenza dell'intero sistema.


**Graph Neural Network (GNN)**
La GNN rappresenta il motore di raccomandazione.
Utilizzando il grafo delle interazioni tra clienti e prodotti, il modello apprende automaticamente i pattern di acquisto e le relazioni latenti presenti nei dati.
Il suo compito è produrre una lista di prodotti consigliati per ciascun cliente, sfruttando la struttura del grafo anziché considerare solamente le informazioni tabellari tradizionali.
L'output della GNN consiste quindi nelle raccomandazioni candidate che verranno successivamente interpretate dall'agente AI.


**Agente AI**
L'agente AI costituisce il livello di interazione con l'utente.
Riceve le richieste formulate in linguaggio naturale e coordina l'utilizzo delle altre componenti del sistema.
In particolare, può:
- interrogare Neo4j per recuperare informazioni contestuali;
- utilizzare le raccomandazioni prodotte dalla GNN;
- combinare le informazioni disponibili per costruire una risposta comprensibile e motivata.

L'obiettivo dell'agente non è solamente mostrare una raccomandazione, ma contestualizzarla e renderla facilmente interpretabile dall'utente finale.


**Flusso**

```text
                +----------------+
                |     Neo4j      |
                | Knowledge Graph|
                +-------+--------+
                        |
                        |
         +--------------+--------------+
         |                             |
         |                             |
+--------v--------+            +-------v--------+
|      GNN        |            |    Agente AI   |
| Recommendation  |----------->| Ragionamento   |
|      Engine     |            | e Interazione  |
+-----------------+            +----------------+
                                       |
                                       |
                                 Risposta finale
```


L'architettura è progettata secondo una logica modulare, nella quale ogni componente svolge un ruolo ben definito:
- **Neo4j** rappresenta la conoscenza.
- **La GNN** apprende dai dati e genera le raccomandazioni.
- **L'agente AI** coordina le informazioni disponibili e produce una risposta interpretabile per l'utente.

Questa separazione delle responsabilità rende il sistema facilmente estendibile, consentendo l'evoluzione indipendente delle singole componenti senza modificare l'intera architettura.
