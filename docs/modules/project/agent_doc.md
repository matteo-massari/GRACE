# Graph RAG

Graph RAG is the conversational layer over the knowledge graph. It receives
a natural-language question, retrieves data from Neo4j and uses an LLM to build the
answer. Recommendations are available only after importing
`RECOMMENDED_TO` relationships; the GNN model is not executed during the application.

## Components of code

The package is `graph-rag-agent/src/graph_rag_agent/`.

| Module | Function |
| --- | --- |
| `agent.py` | Defines `sales_agent`, the commercial prompt and tool `get_data` |
| `tools/subagent.py` | Implement the LangGraph flow `nl2cypher` |
| `prompts/prompts.py` | Contains terminology, examples and prompts to generate Cypher |
| `utils/config.py` | Manages shared Neo4j drivers and LLM clients |
| `graph-rag-agent/main.py` | Read the terminal application and start a request |

## How a question is processed

1. Clients are initialized and read the Neo4j scheme with
   `get_schema(..., is_enhanced=True)`.
2. `sales_agent`, built with OpenAI Agents, receives the question. Your prompt
   requires retrieving evidence through `get_data` for factual questions.
3. The tool starts the subgraph `nl2cypher` with the question.
4. The node `retriever` provides LLM schema, terminology, examples and possible
   previous feedback. Extract Cypher query and run it with `session.execute_read`.
5. The node `annotator` asks LLM to assess the sufficiency of results and
   return a JSON with judgment, motivation and suggestions.
6. The flow ends if the data is judged sufficient, after five attempts
   or when two consecutive queries are identical. Otherwise it repeats recovery.
7. Subgraph status, with results and evaluations, is serialized and
   returned to the agent for the final answer.

```{mermaid}
flowchart TD
    Q["Question"] --> A["sales_agent"]
    A --> T["get_data"]
    T --> R["retriever: generate and execute Cypher"]
    R --> E["annotator: evaluate results"]
    E --> D{"Sufficient data or limit reached?"}
    D -->|No: feedback| R
    D -->|Yes| S["State with evidence and evaluations"]
    S --> A
    A --> F["Final answer"]
```

The commercial prompt requires an answer consisting of result, evidence and
next step, distinguishing available facts and hypotheses. These are instructions
to the model, not a guarantee that every answer is correct. A termination
for limit or repeated query does not certify the sufficiency of data.

## Configuration and execution

From the root of the repository, after configuring `.env`:

```bash
make synthetic-data
make start-neo4j
make pipeline-synthetic
make agent ARGS="Which customers in Roma have the most purchases?"
```

For questions about GNN recommendations, complete the path first
training, inference and `make pipeline-recommendations` of the
[operational guide](../getting_started.md).

Required `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`,
`AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`. The client uses the endpoint as
base URL, then it must be compatible with the API called.

The identifications currently written in the code are `gpt-5.6-luna` for
the main agent and `gpt-5.4-mini` for recovery and evaluation. The endpoint must
exposing those models, or identifiers must be adequate in the code. The
their availability is not verified by filling out the documentation.

`make agent ARGS="--help"` does not open connections. A real question requires
services and can make multiple calls LLM. The program ends after the answer:
no HTTP server, web chat or persistent session are implemented here.
Port variables in the configuration do not start such services.

## Implementation boundaries

- Recovery uses Cypher query on the graph; there is no vector search path
  or an active Qdrant index in this flow.
- Control `is_probably_cypher` search keywords: it is not a parser nor a
  full query validation. Actual permissions depend on the user Neo4j.
- Recovery errors are reported in the state or JSON returned by the tool.
- Recognition of a seller described in the prompt does not constitute a
  user authentication system.
- LLM responses were not verified live in the control session
  local; see the [report](../../verification.md).
