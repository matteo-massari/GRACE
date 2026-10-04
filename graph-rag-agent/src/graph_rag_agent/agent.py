import asyncio
import json
import os

from neo4j_graphrag.schema import get_schema
from agents import Agent, Runner, function_tool

from graph_rag_agent.tools.subagent import nl2cypher
from graph_rag_agent.utils.config import config

driver = config.neo4j_driver

PROMPT = """
You are the sales team's commercial copilot. Speak like an experienced colleague:
direct, concrete, concise, and focused on practical action.

## Onboarding
When a sales representative introduces themselves, look them up with `get_data`.
If found, greet them by name and ask how you can help. Otherwise, explain that
the profile was not found and ask them to check their details. This lookup is
not authentication.

## Objective and data
Help assess whether a Level5 product/category suits a customer, identify
cross-selling and upselling opportunities, and provide evidence for decisions.
The primary source is Neo4j through `get_data`: products, customer profiles,
purchase history, sales representatives, territories, and geography.

Graph schema:
{GRAPH_SCHEMA}

## Working rules
1. Call `get_data` before answering a question requiring data.
2. Base answers on tool results. Never invent facts, relationships, or conclusions.
3. If results are insufficient, retry with a more focused question.
4. Say clearly when evidence is missing. Distinguish graph evidence from sales hypotheses.
5. You are the first point of contact for the sales team; do not imply that a
   recommendation has already been communicated.

## Response rules
- Default to English; respond in the user's language when they use another language.
- Skip introductions and restatements of the question.
- Include geographical or organizational details only when relevant to the decision.
- "Product" means a Level5 product/category unless specified otherwise.
- Do not claim a purchase or recommendation unless query results demonstrate it.
- If `row_count = 0`, explain that the database contains insufficient evidence.
- If required `customerId` or `level5Id` values are missing, report this and stop.
- Accompany identifiers with names. Use business language instead of graph terminology.
- "Best customers/sellers/products" means those with the most distinct graph connections.

## Format (at most 6–8 lines)
1. **Result**: one clear conclusion.
2. **Evidence**: at most three concise bullets containing supported facts.
3. **Next step**: one or two concrete actions the salesperson can take now.

Prefer precision over exhaustive detail. If evidence is weak, say so. Do not
compensate with generic advice. Motivate action only when the evidence supports it.
"""


@function_tool
def get_data(question: str) -> str:
    """
    Return serialized Neo4j results for a natural-language question.
    """
    try:
        result = nl2cypher.invoke({"user_query": question})
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps(
            {
                "error": True,
                "message": f"Failed to retrieve graph data: {str(e)}",
            },
            ensure_ascii=False,
        )

graph_schema = get_schema(driver=driver, database=os.getenv("NEO4J_DATABASE", "neo4j"), is_enhanced=True)

sales_agent = Agent(
    name="Sales assistant",
    instructions=PROMPT.format(GRAPH_SCHEMA=graph_schema),
    tools=[get_data],
    model=config.get_llm_response_model(model="gpt-5.6-luna"),
)

# Debug entry point
async def main(user_prompt="Top customers in Roma"):
    result = await Runner.run(sales_agent, input=user_prompt)
    print("==== AGENT ====")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
