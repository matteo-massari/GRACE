# Graph Loader

```{automodule} graph_recc_sys.graph_loader
:members:
:show-inheritance:
```


## Complete sampler for demo

`--sampler full` uses all nodes and limits the context to previous edges
first event of each batch. The original sampler `neighbor` requires `pyg-lib`.
The two modes share split and pattern but not the same sampling.

```{automodule} graph_recc_sys.full_graph_loader
:members:
:show-inheritance:
```
