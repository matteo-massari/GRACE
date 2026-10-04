# Graph Recommendation System

Il progetto unisce tre componenti: **Knowledge Graph e pipeline**, **rete neurale
GNN** e **Graph RAG**. Neo4j conserva il dominio commerciale; la GNN genera
raccomandazioni dai CSV; Graph RAG interroga Neo4j e formula risposte con un LLM.
MLflow è il servizio di supporto per tracciare training e modelli.

Per iniziare, seguire la [guida operativa](modules/getting_started.md).
La [panoramica dell'architettura](modules/project.md) mostra i collegamenti
attualmente implementati. Le pagine dei componenti descrivono dati, algoritmo e
limiti; la [verifica locale](verification.md) riporta ciò che è stato provato.

```{toctree}
:maxdepth: 1
:caption: Documentazione

Avvio e configurazione <modules/getting_started>
Le tre componenti <modules/project>
Mappa del codice e API <modules/code>
Teoria generale <modules/theory>
Esperimenti <modules/experiments>
Verifiche eseguite <verification>
```

Gli HTML sono generati dai sorgenti Markdown, reStructuredText e dalle docstring
in `docs/` con `make docs`. Non vanno modificati direttamente. La cartella
`docs/_build/` è esclusa da Git; `make full-docs` avvia la consultazione locale
in entrambe le lingue; ricompilare dopo ogni modifica.

```{toctree}
:hidden:
:caption: Riferimenti e archivio

modules/modules
archive/gnn_agent_history
archive/gnn_model_history
```
