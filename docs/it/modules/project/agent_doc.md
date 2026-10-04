# Graph RAG

Graph RAG è il livello di consultazione commerciale del Knowledge Graph. Riceve
una domanda in italiano, recupera dati da Neo4j e usa un LLM per costruire la
risposta. Le raccomandazioni sono disponibili soltanto dopo l'importazione delle
relazioni `RECOMMENDED_TO`; il modello GNN non viene eseguito durante la domanda.

## Componenti del codice

Il package è `graph-rag-agent/src/graph_rag_agent/`.

| Modulo | Funzione |
| --- | --- |
| `agent.py` | Definisce `sales_agent`, il prompt commerciale e il tool `get_data` |
| `tools/subagent.py` | Implementa il flusso LangGraph `nl2cypher` |
| `prompts/prompts.py` | Contiene terminologia, esempi e prompt per generare Cypher |
| `utils/config.py` | Gestisce driver Neo4j e client LLM condivisi |
| `graph-rag-agent/main.py` | Legge la domanda da terminale e avvia una richiesta |

## Come viene elaborata una domanda

1. All'avvio vengono inizializzati i client e letto lo schema Neo4j con
   `get_schema(..., is_enhanced=True)`.
2. `sales_agent`, costruito con OpenAI Agents, riceve la domanda. Il suo prompt
   richiede il recupero di dati tramite `get_data` per le risposte fattuali.
3. Il tool avvia il sottografo `nl2cypher` con la domanda.
4. Il nodo `retriever` fornisce al LLM schema, terminologia, esempi ed eventuale
   feedback precedente. Estrae la query Cypher e la esegue con `session.execute_read`.
5. Il nodo `annotator` chiede al LLM di valutare la sufficienza dei risultati e
   restituire un JSON con giudizio, motivazione e suggerimenti.
6. Il flusso termina se i dati sono giudicati sufficienti, dopo cinque tentativi
   oppure quando due query consecutive sono identiche. Altrimenti ripete il recupero.
7. Lo stato del sottografo, con risultati e valutazioni, viene serializzato e
   restituito all'agente per la risposta finale.

```{mermaid}
flowchart TD
    Q["Domanda"] --> A["sales_agent"]
    A --> T["get_data"]
    T --> R["retriever: genera ed esegue Cypher"]
    R --> E["annotator: valuta risultati"]
    E --> D{"Dati sufficienti o limite raggiunto?"}
    D -->|No: feedback| R
    D -->|Sì| S["Stato con evidenze e valutazioni"]
    S --> A
    A --> F["Risposta finale"]
```

Il prompt commerciale richiede una risposta composta da risultato, evidenze e
prossimo passo, distinguendo fatti disponibili e ipotesi. Queste sono istruzioni
al modello, non una garanzia che ogni risposta sia corretta. Una terminazione
per limite o query ripetuta non certifica la sufficienza dei dati.

## Configurazione ed esecuzione

Dalla radice della repository, dopo aver configurato `.env`:

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
make agent ARGS="Quali sono i clienti di Roma con più acquisti?"
```

Per domande sulle raccomandazioni della GNN, completare prima il percorso di
training, inferenza e `make pipeline-recommendations` della
[guida operativa](../getting_started.md).

Sono necessari `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`,
`AZURE_OPENAI_API_KEY` e `AZURE_OPENAI_ENDPOINT`. Il client usa l'endpoint come
base URL, quindi deve essere compatibile con le API chiamate.

Gli identificativi attualmente scritti nel codice sono `gpt-5.6-luna` per
l'agente principale e `gpt-5.4-mini` per recupero e valutazione. L'endpoint deve
esporre quei modelli, oppure gli identificativi vanno adeguati nel codice. La
loro disponibilità non viene verificata dalla compilazione della documentazione.

`make agent ARGS="--help"` non apre connessioni. Una domanda reale richiede i
servizi e può effettuare più chiamate LLM. Il programma termina dopo la risposta:
non sono implementati qui un server HTTP, una chat web o una sessione persistente.
Le variabili di porta presenti nella configurazione non avviano tali servizi.

## Confini dell'implementazione

- Il recupero usa query Cypher sul grafo; non c'è un percorso di ricerca vettoriale
  o un indice Qdrant attivo in questo flusso.
- Il controllo `is_probably_cypher` cerca parole chiave: non è un parser né una
  validazione completa delle query. I permessi effettivi dipendono dall'utente Neo4j.
- Gli errori di recupero vengono riportati nello stato o nel JSON restituito dal tool.
- Il riconoscimento di un venditore descritto nel prompt non costituisce un
  sistema di autenticazione degli utenti.
- Le risposte LLM non sono state verificate dal vivo nella sessione di controllo
  locale; vedere il [resoconto](../../verification.md).
