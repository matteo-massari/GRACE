# Assistente Graph RAG
[English version](README.md)

L'assistente usa OpenAI Agents per rispondere alle domande commerciali e un
flusso LangGraph per generare, eseguire e verificare interrogazioni Cypher.
Il codice originale è nel package Python `src/graph_rag_agent/`.

Dalla radice della repository:

```bash
uv sync --locked
make synthetic-data
make start-neo4j
make pipeline-synthetic
make agent ARGS="Quali sono i clienti di Roma con più acquisti?"
```

Configurare `.env` dall'esempio alla radice: `NEO4J_URI`, `NEO4J_USERNAME`,
`NEO4J_PASSWORD`, `NEO4J_DATABASE`, `AZURE_OPENAI_API_KEY` e
`AZURE_OPENAI_ENDPOINT`. L'endpoint deve esporre le API compatibili con i client
usati e i modelli indicati in `agent.py` e `tools/subagent.py`.

`make agent ARGS="--help"` mostra l'utilizzo senza collegarsi ai servizi.
Una domanda reale richiede Neo4j e un endpoint LLM configurati; può consumare
credito API. Il caricamento di dati sintetici non richiede il servizio LLM.

## Flusso e documentazione

L'agente principale chiama `get_data`, che avvia un ciclo di generazione Cypher,
interrogazione Neo4j e valutazione LLM. Il ciclo termina quando le evidenze sono
ritenute sufficienti, dopo cinque tentativi o su due query consecutive identiche.
L'agente formula quindi la risposta. Non interroga direttamente la rete GNN:
usa le raccomandazioni già importate in Neo4j.

L'entry point esegue una domanda e termina; non avvia un'interfaccia web.
Gli identificativi dei modelli presenti nel codice sono `gpt-5.6-luna` per
l'agente principale e `gpt-5.4-mini` per recupero e valutazione: devono essere
disponibili sull'endpoint configurato o adeguati ai deployment accessibili.

- [Funzionamento, stato e limiti dell'agente](../docs/it/modules/project/agent_doc.md)
- [Guida operativa delle tre componenti](../docs/it/modules/getting_started.md)
- [Verifiche eseguite](../docs/it/verification.md)
