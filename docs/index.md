# GRACE

GRACE combines a **Neo4j knowledge graph and data pipeline**, a **graph neural
recommendation model**, and a **Graph RAG assistant**. The GNN learns from local
CSV data; the assistant retrieves Neo4j evidence and answers with an LLM.
MLflow tracks training runs and model artifacts.

Start with the [operational guide](modules/getting_started.md) and
[architecture overview](modules/project.md). Component pages explain the data,
algorithms, and limitations. The [verification report](verification.md) records
which checks have been performed.

```{toctree}
:maxdepth: 1
:caption: Documentation

Start and configuration <modules/getting_started>
The three components <modules/project>
Code Map and API <modules/code>
General Theory <modules/theory>
Experiments <modules/experiments>
Verifications performed <verification>
```

Build the site with `make docs`. English sources live in `docs/`; Italian prose
lives in `docs/it/`. The language switch preserves the current page. Theme
configuration, API docstrings, and common images are shared. Generated HTML in
`docs/_build/` is ignored by Git. Run `make full-docs` to build and serve both languages;
rebuild after editing the sources.

```{toctree}
:hidden:
:caption: References and archive

modules/modules
archive/gnn_agent_history
archive/gnn_model_history
```
