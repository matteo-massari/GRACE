"""Configurazione condivisa del Graph RAG agent."""

import atexit
import logging
import os

from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel, init_chat_model
from neo4j import Driver, GraphDatabase
from openai import AsyncOpenAI, OpenAI


class Config:
    """Singleton che raccoglie configurazione e client condivisi dell'agente."""

    _instance: "Config | None" = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        load_dotenv()

        self._logger = logging.getLogger("graph_rag_agent")
        self._host = os.getenv("HOST", "0.0.0.0")
        self.AGENT_SERVER_PORT = int(os.getenv("AGENT_SERVER_PORT", "8000"))
        self.CHAINLIT_CLIENT_PORT = int(os.getenv("CHAINLIT_CLIENT_PORT", "3546"))

        self._openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self._openai_endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://3114-azure-openai-api.openai.azure.com/openai/v1/",
        )
        self._neo4j_uri = os.getenv("NEO4J_URI")
        self._neo4j_username = os.getenv("NEO4J_USERNAME")
        self._neo4j_password = os.getenv("NEO4J_PASSWORD")

        self._sync_openai_client: OpenAI | None = None
        self._async_openai_client: AsyncOpenAI | None = None
        self._llm_clients: dict[tuple[str, str], BaseChatModel] = {}
        self._neo4j_driver: Driver | None = None
        atexit.register(self._shutdown)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def _require_openai_config(self) -> tuple[str, str]:
        if not self._openai_api_key or not self._openai_endpoint:
            raise ValueError(
                "OpenAI API key or endpoint not configured properly. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )
        return self._openai_api_key, self._openai_endpoint

    def get_openai_client(self, async_: bool = True) -> OpenAI | AsyncOpenAI:
        """Restituisce il client OpenAI sincrono o asincrono condiviso."""
        api_key, base_url = self._require_openai_config()
        if async_:
            if self._async_openai_client is None:
                self._async_openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                self.logger.info("OpenAI async client created")
            return self._async_openai_client

        if self._sync_openai_client is None:
            self._sync_openai_client = OpenAI(api_key=api_key, base_url=base_url)
            self.logger.info("OpenAI sync client created")
        return self._sync_openai_client

    def get_llm_client(self, model: str, client: str) -> BaseChatModel:
        """Crea e mette in cache un modello LangChain OpenAI."""
        if client != "openai":
            raise ValueError(f"Client/provider not supported: {client}")

        key = (client, model)
        if key not in self._llm_clients:
            api_key, base_url = self._require_openai_config()
            self._llm_clients[key] = init_chat_model(
                model=model,
                model_provider="openai",
                api_key=api_key,
                base_url=base_url,
            )
            self.logger.info("LLM client created: provider=%s, model=%s", client, model)
        return self._llm_clients[key]

    def get_llm_response_model(self, model: str) -> OpenAIResponsesModel:
        """Crea un modello Responses per OpenAI Agents."""
        return OpenAIResponsesModel(model=model, openai_client=self.get_openai_client(async_=True))

    def get_llm_chat_completions_model(self, model: str) -> OpenAIChatCompletionsModel:
        """Crea un modello Chat Completions per OpenAI Agents."""
        return OpenAIChatCompletionsModel(
            model=model,
            openai_client=self.get_openai_client(async_=True),
        )

    @property
    def neo4j_driver(self) -> Driver:
        """Inizializza lazy e restituisce il driver Neo4j condiviso."""
        if self._neo4j_driver is None:
            if not all((self._neo4j_uri, self._neo4j_username, self._neo4j_password)):
                raise ValueError(
                    "Missing Neo4j environment variables: "
                    "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
                )
            self._neo4j_driver = GraphDatabase.driver(
                self._neo4j_uri,
                auth=(self._neo4j_username, self._neo4j_password),
            )
            self._neo4j_driver.verify_connectivity()
            self.logger.info("Neo4j driver created and connection verified")
        return self._neo4j_driver

    def get_agent_server_port(self) -> int:
        return self.AGENT_SERVER_PORT

    def get_host(self) -> str:
        return self._host

    def _shutdown(self) -> None:
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()
            self._neo4j_driver = None


config = Config()
