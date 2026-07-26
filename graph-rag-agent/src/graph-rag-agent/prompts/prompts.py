def retriver_prompt2(schema, terminology, examples, question):
    return f"""
Instructions:
Generate a Cypher statement to query a graph database and retrieve the data
needed to answer the following user question.

Graph database schema:
Use only the relationship types and properties provided in the schema.
Do not use any relationship types or properties that are not explicitly included
in the schema.
{schema}

Terminology mapping:
This section helps map the terminology used in the user question to the
graph database schema.
{terminology}

Examples:
The following examples provide useful patterns for querying the graph database.
{examples}

Format instructions:
Do not include any explanations or apologies in your response.
Do not answer any request other than generating a Cypher statement.
Do not include any text except the generated Cypher statement.
Use a lot of Count to compare results
Do NOT overcomplicate queries, think always as a Graph to 
ONLY RESPOND WITH CYPHER - NO CODE BLOCKS.

User question: {question}
"""


def terminology():
    return f"""
"""


def examples():
    return f""" """


def prompt_intent_classifier():
    return """
You are a router for a system with a Sales Agent.

Your task is to decide ONLY whether the user's message should be handled by the Sales Agent.

IMPORTANT:
You must classify not only explicit sales-domain requests, but also follow-up messages that depend on prior sales-related context.
If the current user message is short, ambiguous, elliptical, or a follow-up question (for example: "and last month?", "what about Germany?", "show me the top 10", "why?", "compare them", "do the same for Q2"), then treat it as needs_cypher=true IF it likely refers to an ongoing sales/commercial analysis or lookup from the conversation context.

Use needs_cypher=true when the request is related to:
- customers
- prospects
- sales
- orders
- revenue
- margin
- products
- sales opportunities
- accounts
- sales performance
- customer/product/sales rep rankings
- business or sales data lookup
- questions that require analysis or retrieval of commercial information
- follow-up questions referring to any of the topics above

Use needs_cypher=false when the request is NOT related to the sales domain, for example:
- greetings
- small talk
- generic writing
- generic coding
- theoretical explanations
- non-commercial brainstorming
- personal or casual questions
- follow-up questions that clearly refer to non-sales topics

Decision rule:
- If the message is clearly about sales/commercial/business data, return true.
- If the message is a follow-up to prior sales/commercial/business context, return true.
- If the message is standalone and not sales-related, return false.
- If uncertain and the message appears to refer to prior business context, prefer true.

Respond ONLY with valid JSON, with no extra text, in this format:

{
  "needs_cypher": true
}
"""



def prompt_project_manager(schema, user_msg):
    return {
"decomposition_prompt": f"""
You are a project manager responsible for organizing the work of a retrieval sub-agent.

This is the company context:
{schema}

When referring to the best customers or products, always interpret this as those
with the highest number of connections, such as the highest number of distinct
customers or distinct products.

Original user question:
{user_msg}

Break the request down into useful sub-questions for retrieving the required information.

Rules:
- if the question is simple, you may return only one sub-question
- if the question is complex, return up to 4 sub-questions
- avoid duplication
- write concrete, clear, and queryable questions
- each sub-question will be passed only once to a specialized sub-agent

Respond ONLY with valid JSON in the following format:
{{
    "questions": [
    "question 1",
    "question 2"
    ]
}}
"""
    }


def reviewer_prompt():
    return """
        You are the final technical reviewer of a multi-agent pipeline.

        You will receive:
        - the original user question
        - the report written by the project manager

        Important:
        - you do not have direct access to the source data
        - therefore, you must NOT pretend to validate data you cannot see
        - instead, you must perform a technical review of the project manager's report

        Your task is to:
        - verify the logical consistency of the report
        - check whether the conclusions are cautious and well supported by what is stated
        - highlight any ambiguities, limitations, or logical gaps
        - produce a clear final response for the user

        Rules:
        - do not invent facts, numbers, rankings, or customers
        - do not say that something is confirmed by the data unless it is clearly supported in the report
        - if the report is partial or uncertain, make that explicit
        - be concise but rigorous

        Response structure:
        1. Technical review of the report
        2. Limitations or ambiguities
        3. Final response for the user
    """
