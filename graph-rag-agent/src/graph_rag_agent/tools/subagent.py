import json
import os
import re
from operator import add
from typing import Annotated, Any, Dict, List, Optional

from neo4j import Neo4jDriver
from neo4j_graphrag.schema import get_schema
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from graph_rag_agent.prompts.prompts import examples, retriver_prompt2, terminology
from graph_rag_agent.utils.config import config
from langchain.agents import create_agent

driver: Neo4jDriver = config.neo4j_driver
llm = config.get_llm_client(model="gpt-5.4-mini", client="openai")
logger = config.logger


def get_database_name() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


class SubgraphState(BaseModel):
    # Local state tracks subgraph progress
    user_query: str
    iteration: int = 0
    max_iterations: int = 5
    done: bool = False

    # Conversation state retains the message history.
    annotation: Annotated[List[Dict[str, Any]], add] = Field(default_factory=list)
    retrieved: Annotated[List[Dict[str, Any]], add] = Field(default_factory=list)


# Extract Cypher queries
def extract_cypher(text: str) -> str:
    if not text:
        return ""

    pattern = r"```(?:[a-zA-Z]+\n)?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()

    return text.strip()



def is_probably_cypher(text: str) -> bool:
    if not text:
        return False

    cypher_keywords = [
        "MATCH",
        "RETURN",
        "WITH",
        "OPTIONAL MATCH",
        "CALL",
        "UNWIND",
    ]
    upper_text = text.upper()
    return any(keyword in upper_text for keyword in cypher_keywords)

# Execute Cypher
def run_cypher_retrieval(
    driver: Neo4jDriver,
    database: str,
    cypher_query: str,
) -> List[Dict[str, Any]]:
    def run_read_query(tx, query: str):
        result = tx.run(query)
        return [record.data() for record in result]

    with driver.session(database=database) as session:
        records = session.execute_read(run_read_query, cypher_query)

    return records

# Fetch the schema
def get_graph_schema(database: str) -> str:
    try:
        return get_schema(driver=driver, database=database, is_enhanced=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Neo4j schema: {str(e)}") from e


def build_retriever_prompt(
    schema: str,
    user_query: str,
    previous_annotation: Optional[Dict[str, Any]] = None,
    previous_retrieval: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = retriver_prompt2(
        schema=schema,
        terminology=terminology(),
        examples=examples(),
        question=user_query,
    )

    if previous_annotation:
        feedback = previous_annotation.get("feedback_for_retriever", "")
        prompt += f"""

Additional instructions based on the previous retrieval attempt:
Original user question:
{user_query}

Previous feedback from evaluator:
{feedback}
"""

        if previous_retrieval:
            prompt += f"""
Previous Cypher query:
{previous_retrieval.get('cypher', '')}

Previous query results:
{json.dumps(previous_retrieval.get('results', []), ensure_ascii=False, indent=2, default=str)}

Previous execution error:
{previous_retrieval.get('error')}
"""

        prompt += """
Please improve the Cypher query accordingly.
Return only the Cypher query, with no code block and no extra text.
"""

    return prompt

# The retriever generates a Cypher query
def retriever(state: SubgraphState):
    database = get_database_name()
    schema = get_graph_schema(database)

    previous_annotation = state.annotation[-1] if state.annotation else None
    previous_retrieval = state.retrieved[-1] if state.retrieved else None

    prompt = build_retriever_prompt(
        schema=schema,
        user_query=state.user_query,
        previous_annotation=previous_annotation,
        previous_retrieval=previous_retrieval,
    )

    query_response = llm.invoke(prompt).content
    cypher_query = extract_cypher(query_response)

    retrieval_error = None
    records: List[Dict[str, Any]] = []

    if not cypher_query or not is_probably_cypher(cypher_query):
        retrieval_error = "LLM did not return a valid Cypher query."
    else:
        try:
            records = run_cypher_retrieval(
                driver=driver,
                database=database,
                cypher_query=cypher_query,
            )
        except Exception as e:
            retrieval_error = str(e)

    logger.info(f"Cypher query: {cypher_query}")
    if retrieval_error:
        logger.warning(f"Retrieval error: {retrieval_error}")

    retrieval_output = {
        "iteration": state.iteration + 1,
        "user_query": state.user_query,
        "cypher": cypher_query,
        "results": records,
        "error": retrieval_error,
        "feedback_used": previous_annotation.get("feedback_for_retriever") if previous_annotation else None,
    }

    return {
        "retrieved": [retrieval_output],
        "iteration": state.iteration + 1,
    }


def build_evaluation_prompt(
    user_query: str,
    schema: str,
    last_retrieval: Dict[str, Any],
) -> str:
    return f"""
You are an evaluator for a graph retrieval system.

User question:
{user_query}

Graph schema:
{schema}

Executed Cypher query:
{last_retrieval.get('cypher')}

Retrieved results:
{json.dumps(last_retrieval.get('results', []), ensure_ascii=False, indent=2, default=str)}

Execution error:
{last_retrieval.get('error')}

Your task:
1. Decide whether the retrieved information is sufficient to answer the user's question.
2. If sufficient, set is_sufficient=true.
3. If not sufficient, set is_sufficient=false and provide precise feedback to improve the next Cypher query.
4. If the query returned no results, do not assume it is correct.
5. Be critical about:
- relevance
- completeness
- correct geographic or entity filtering
- whether the query is too broad or too narrow
- whether ranking or aggregation is missing
- whether labels, relationships, properties, or filters may be incorrect
- if the query is empty, invalid, or returns no rows, suggest concrete corrections
- if a city filter may be the problem, suggest checking variants such as ROMA, Roma, or roma only as one possible hypothesis, not as the only fix

Return ONLY valid JSON in this format:
{{
  "is_sufficient": true,
  "reason": "short explanation",
  "feedback_for_retriever": "specific instruction for the retriever"
}}
"""


def annotator(state: SubgraphState):
    if not state.retrieved:
        annotation = {
            "iteration": state.iteration,
            "is_sufficient": False,
            "reason": "No retrieval results available to evaluate.",
            "feedback_for_retriever": "Generate a valid Cypher query that answers the user question.",
        }
        return {
            "annotation": [annotation],
            "done": False,
        }

    database = get_database_name()
    schema = get_graph_schema(database)
    last_retrieval = state.retrieved[-1]

    evaluation_prompt = build_evaluation_prompt(
        user_query=state.user_query,
        schema=schema,
        last_retrieval=last_retrieval,
    )

    llm_response = llm.invoke(evaluation_prompt).content

    try:
        evaluation = json.loads(llm_response)
    except json.JSONDecodeError:
        evaluation = {
            "is_sufficient": False,
            "reason": "Evaluator did not return valid JSON.",
            "feedback_for_retriever": "Return valid JSON and improve the Cypher query to better answer the user question.",
        }

    annotation = {
        "iteration": state.iteration,
        "is_sufficient": evaluation.get("is_sufficient", False),
        "reason": evaluation.get("reason", ""),
        "feedback_for_retriever": evaluation.get("feedback_for_retriever", ""),
    }

    return {
        "annotation": [annotation],
        "done": annotation["is_sufficient"],
    }


def should_retrieve_more(state: SubgraphState):
    if state.done:
        return END

    if state.iteration >= state.max_iterations:
        return END

    if len(state.retrieved) >= 2:
        last_query = state.retrieved[-1].get("cypher", "").strip()
        prev_query = state.retrieved[-2].get("cypher", "").strip()
        if last_query and last_query == prev_query:
            return END

    return "retriever"


builder = StateGraph(SubgraphState)
builder.add_node("retriever", retriever)
builder.add_node("annotator", annotator)

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "annotator")
builder.add_conditional_edges(
    "annotator",
    should_retrieve_more,
    {
        "retriever": "retriever",
        END: END,
    },
)

nl2cypher = builder.compile()

if __name__ == "__main__":
    result = nl2cypher.invoke({"user_query": "10 customers in Roma with the most distinct purchases"})
    print(result["retrieved"][-1]["results"])
