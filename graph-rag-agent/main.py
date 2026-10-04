"""Command-line entry point; --help does not connect to external services."""

import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(description="Graph RAG sales assistant")
    parser.add_argument("question", help="Question to ask the agent")
    args = parser.parse_args()
    from graph_rag_agent.agent import main as run_agent

    asyncio.run(run_agent(args.question))


if __name__ == "__main__":
    main()
