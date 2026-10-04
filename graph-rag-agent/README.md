# Graph RAG assistant

**English** | [Italiano](README.it.md)

The assistant uses OpenAI Agents and a LangGraph workflow to generate Cypher,
retrieve Neo4j records, and evaluate whether the results answer the user's question.
It defaults to English and follows the user's language.

From the repository root:

```bash
uv sync --locked
make synthetic-data
make start-neo4j
make pipeline-synthetic
make agent ARGS="Which customers in Roma have the most purchases?"
```

Configure `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`,
`AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_ENDPOINT` in `.env`. The endpoint must
support the APIs and model identifiers configured in `agent.py` and
`tools/subagent.py`. The current identifiers are `gpt-5.6-luna` and `gpt-5.4-mini`;
adjust them to match your available deployments.

`make agent ARGS="--help"` displays usage without connecting to services.
A real question requires Neo4j and LLM access. Loading synthetic data needs no LLM.

The agent calls `get_data`; retrieval stops when evidence is judged sufficient,
after five attempts, or after two identical consecutive queries. It then produces
a response. Recommendations come from Neo4j after import, so complete GNN training,
inference, and `make pipeline-recommendations` before asking about model output.
The command handles one question and exits.

- [Agent architecture and limitations](../docs/modules/project/agent_doc.md)
- [End-to-end guide](../docs/modules/getting_started.md)
- [Verification report](../docs/verification.md)
