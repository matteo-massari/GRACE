# Mappa del codice e API

La repository contiene tre componenti applicativi. Le pagine API sottostanti
sono generate dalle docstring della GNN; pipeline e Graph RAG sono descritti
anche nei rispettivi capitoli, senza importarli durante la build e aprire
connessioni ai servizi.

## Knowledge Graph e pipeline

| File in `pipeline/` | Responsabilità |
| --- | --- |
| `generate_synthetic_database.py` | Genera CSV deterministici per Neo4j e GNN |
| `load_synthetic_neo4j.py` | Crea vincoli, nodi e relazioni da CSV |
| `ETL_recc.py` | Importa le raccomandazioni su clienti e prodotti esistenti |
| `ETL_fact.py` | Conserva il template Impala, non operativo senza integrazione privata |
| `config.py` | Configura la connessione Neo4j |

Vedere [Knowledge Graph e pipeline](project/graph_doc.md).

## Rete neurale

| Classe/modulo in `graph_recc_sys` | Responsabilità |
| --- | --- |
| `DatasetManager` | Percorsi CSV e integrazione opzionale con l'estrazione privata |
| `MarketBasketGraphBuilder` | Grafo eterogeneo, feature, mapping e cache |
| `GraphLoader` | Split temporali, loader e insiemi di esclusione |
| `FullGraphLinkLoader` | Batch temporali completi per piccoli dataset |
| `RecommenderModel` | Embedding, GraphSAGE e proiezioni |
| `NeuralNetworkEngine` | Coordinamento di training, validazione, test e export |
| `Training`, `Inference`, `Evaluation` | Loss, ranking e metriche |
| `HyperParams`, `MlflowConfig` | Configurazione YAML e tracking |
| `__main__` | Opzioni CLI e avvio della pipeline |

```{toctree}
:maxdepth: 2
:caption: API GNN

Dataset ed estrazione <code/extraction>
Costruzione del grafo <code/graph_dataset>
Split e loader <code/graph_loader>
Modello <code/neural_network>
Engine, training e valutazione <code/engine>
Configurazione e tracking <code/configuration>
Estrazione SQL privata <code/query>
```

## Graph RAG

`graph_rag_agent.agent` contiene l'agente e il tool `get_data`;
`graph_rag_agent.tools.subagent` contiene i nodi `retriever` e `annotator` del
flusso LangGraph. `prompts` raccoglie esempi e terminologia, `utils.config` i
client. L'entry point è `graph-rag-agent/main.py`.

Vedere [flusso e configurazione Graph RAG](project/agent_doc.md).
