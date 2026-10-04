# Graph Loader

```{automodule} graph_recc_sys.graph_loader
:members:
:show-inheritance:
```


## Sampler completo per la demo

`--sampler full` usa tutti i nodi e limita il contesto agli archi precedenti al
primo evento di ogni batch. Il sampler originale `neighbor` richiede `pyg-lib`.
Le due modalità condividono split e modello ma non lo stesso campionamento.

```{automodule} graph_recc_sys.full_graph_loader
:members:
:show-inheritance:
```
