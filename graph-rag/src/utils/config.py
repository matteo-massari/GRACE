import os

from neo4j import Driver, GraphDatabase

import atexit
import os

from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from dotenv import load_dotenv
from google import genai
from langchain.chat_models import BaseChatModel, init_chat_model
from openai import AsyncOpenAI, OpenAI
from qdrant_client import QdrantClient

# TODO: Mettere il logger

class BaseConfig:
	_instance = None
	_initialized = False

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	def __init__(self):
		if self._initialized:
			return
		# inizializzazione
		self._initialized = True

		load_dotenv()

		# fast api var
		self.__AGENT_SERVER_PORT = int(os.getenv("AGENT_SERVER_PORT", 8000))
		self.__HOST = os.getenv("HOST", "0.0.0.0")

		# key
		self.__OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
		self.__GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

		# OpenAI Endpoint
		self.__OPENAI_ENDPOINT = os.getenv(
			"AZURE_OPENAI_ENDPOINT", "https://3114-azure-openai-api.openai.azure.com/openai/v1/"
		)


		# Redis

		# Cache client
		self._logger = "Aggiungere il logger"
		self._sync_openai_client = None  # client OpenAI sync
		self._async_openai_client = None  # client OpenAI async
		self._google_client = None  # client di google
		self._llm_client = {}  # client specifico per langchain
		self._vectordb = None
		self._redis_client = None

		if self.__REDIS_USE:
			self.get_redis_client()

		atexit.register(self._shutdown)

	@staticmethod
	def _parse_bool_env(value: str | None, default: bool) -> bool:
		"""Parse boolean env values in a permissive and predictable way."""
		if value is None:
			return default
		return value.strip().lower() in {
			"1",
			"true",
			"True",
			"yes",
			"y",
			"on",
			"enable",
		}

	def _get_logger_config(
		self,
		*,
		agent_name: str = "assistant",
		default_level: str = "INFO",
		default_use_colors: bool = True,
	) -> dict[str, str | bool]:
		"""Centralized logger configuration sourced from env with sane defaults."""
		# Normalize early so map_level() receives stable uppercase values.
		level = (os.getenv("LOG_LEVEL", default_level) or default_level).strip().upper()
		return {
			"level": level,
			"agent_name": os.getenv("LOGGER_AGENT_NAME", agent_name),
			"use_colors": self._parse_bool_env(os.getenv("LOG_USE_COLORS"), default_use_colors),
		}
		# Logger

	@property
	def logger(self):
		"""
		This method provides access to the logger instantiated in logger.py
		"""
		return self._logger

	# Client per openai
	## OpenAi Client
	@property
	def _sync_openai_client_runner(self):
		if self._sync_openai_client is None:
			if not self.__OPENAI_API_KEY or not self.__OPENAI_ENDPOINT:
				self.logger.error("OpenAI API key or endpoint not configured properly.")
				raise ValueError("OpenAI API key or endpoint not configured properly.")
			self._sync_openai_client = OpenAI(
				api_key=self.__OPENAI_API_KEY,
				base_url=self.__OPENAI_ENDPOINT,
			)
			self.logger.success("OpenAI sync client created")
		return self._sync_openai_client

	@property
	def _async_openai_client_runner(self):
		if not self.__OPENAI_API_KEY or not self.__OPENAI_ENDPOINT:
			self.logger.error("OpenAI API key or endpoint not configured properly.")
			raise ValueError("OpenAI API key or endpoint not configured properly.")
		if self._async_openai_client is None:
			self._async_openai_client = AsyncOpenAI(
				api_key=self.__OPENAI_API_KEY,
				base_url=self.__OPENAI_ENDPOINT,
			)
			self.logger.success("OpenAI async client created")
		return self._async_openai_client

	# Entry point client openAI
	def get_openai_client(self, async_: bool = True) -> OpenAI | AsyncOpenAI:
		"""
		This method provides access to the configured OpenAI client and allows choosing between the synchronous and asynchronous versions.
		"""
		if async_ is True:
			return self._async_openai_client_runner
		else:
			return self._sync_openai_client_runner

	# Client per google
	# entry point per google client, l'utente decide se sincrono o asincorno dopo l'inizializzazione del client
	def get_google_client(self) -> genai.Client:
		"""
		The client is created lazily on first access and then reused.
		"""

		if self._google_client is None:
			if not self.__GOOGLE_API_KEY:
				self.logger.error("Google API key not configured properly.")
				raise ValueError("Google API key not configured properly.")
			self._google_client = genai.Client(
				api_key=self.__GOOGLE_API_KEY,
			)
			self.logger.success("Google GenAI client created")
		return self._google_client

	# Entry point client Langchain
	def get_llm_client(self, model: str, client: str) -> BaseChatModel:
		"""
		This method initializes and caches a LangChain chat model using the specified provider and model name.
		"""
		key = (client, model)

		if key not in self._llm_client:
			try:
				if client == "openai":
					if not self.__OPENAI_API_KEY or not self.__OPENAI_ENDPOINT:
						self.logger.error("OpenAI API key or endpoint not configured properly.")
						raise ValueError("OpenAI API key or endpoint not configured properly.")
					llm = init_chat_model(
						model=model,
						model_provider=client,
						api_key=self.__OPENAI_API_KEY,
						base_url=self.__OPENAI_ENDPOINT,
					)
				elif client == "google_genai":
					if not self.__GOOGLE_API_KEY:
						self.logger.error("Google API key not configured properly.")
						raise ValueError("Google API key not configured properly.")
					llm = init_chat_model(
						model=model,
						model_provider="google_genai",
						api_key=self.__GOOGLE_API_KEY,
					)
				else:
					raise ValueError(f"Client/provider not supported: {client}")

				self.logger.info(f"MODEL={model}, CLIENT = {client}")
				self.logger.success("LLM client created successfully")
				self._llm_client[key] = llm
			except Exception as e:
				self.logger.error(f"LLM client not created correctly: {e}")
				raise
		return self._llm_client[key]

	# Entry Point openaiAgents, puù essere solamente asincrono per costruzione del pachetto openai agents
	def get_llm_response_model(self, model: str) -> OpenAIResponsesModel:
		"""
		This method builds an `OpenAIResponsesModel` configured with the given model name and the asynchronous OpenAI client.
		"""
		try:
			self.logger.info(f"MODEL={model}")
			_model = OpenAIResponsesModel(
				model=model, openai_client=self._async_openai_client_runner
			)
			self.logger.info(f"CLIENT={self._async_openai_client}")
		except Exception as e:
			self.logger.error(f"Model not created correctly: {e}")
			raise
		return _model

	def get_llm_chat_completions_model(self, model: str) -> OpenAIChatCompletionsModel:
		"""
		This method builds an `OpenAIChatCompletionsModel` configured with the given model name and the asynchronous OpenAI client.
		"""
		try:
			self.logger.info(f"MODEL={model}")
			_model = OpenAIChatCompletionsModel(
				model=model, openai_client=self._async_openai_client_runner
			)
			self.logger.info(f"CLIENT={self._async_openai_client}")
		except Exception as e:
			self.logger.error(f"Model not created correctly: {e}")
			raise
		return _model

	# Redis
	def get_redis_use(self) -> bool:
		return self.__REDIS_USE

	def get_redis_url(self) -> str:
		if not self.__REDIS_URL:
			self.logger.error("Redis URL not configured properly.")
			raise ValueError("Redis URL not configured properly.")
		return self.__REDIS_URL

	def get_redis_port(self) -> str:
		return self.__REDIS_PORT

	def get_redis_full_url(self) -> str:
		"""Funzione che ritorna l'url completo del client di redis, da utilizzare per inizializzare il checkpointer su redis di Langgraph"""
		return f"{self.__REDIS_URL}:{self.__REDIS_PORT}"

	def get_redis_ttl_config(self) -> dict:
		"""Funzione che ritorna la configurazione del TTL per il checkpointer di Langgraph"""
		return self.__REDIS_TTL_CONFIG

	def get_redis_client(self):
		"""Funzione che ritorna il client di redis"""
		try:
			import redis

			if not self._redis_client:
				client = redis.Redis(host=self.get_redis_url(), port=self.get_redis_port())

				# Verifica effettiva della connessione
				if client.ping():
					self._redis_client = client
					self.logger.success("Redis client created and connection verified")
				else:
					raise Exception("Redis PING returned False")
				self.logger.success("Redis client created")

			return self._redis_client

		except Exception as e:
			self.logger.error(f"Redis client not created correctly: {e}")
			raise

	# Vector DB
	@property
	def vectordb(self) -> QdrantClient:
		"""
		The client is initialized lazily on first access. A connectivity check is performed by requesting the available collections.
		"""
		if self._vectordb is None:
			try:
				self.logger.info("Connection to qdrant...")
				self._vectordb = QdrantClient(url=self.__QDRANT_URL)
				self._vectordb.get_collections()
				self.logger.info("Success")
			except Exception as e:
				self.logger.error(f"Impossible to create a connection Qdrant: {e}")
				raise
		return self._vectordb

	def get_agent_server_port(self) -> int:
		return self.__AGENT_SERVER_PORT

	def get_host(self) -> str:
		return self.__HOST

	def _shutdown(self):
		if self._vectordb is not None:
			self.logger.info("Shutdown Config")
			self._vectordb.close()



class Config(BaseConfig):
    def __init__(self):
        super().__init__()

        if getattr(self, "_neo4j_initialized", False):
            return

        self._neo4j_uri = os.getenv("NEO4J_URI")
        self._neo4j_username = os.getenv("NEO4J_USERNAME")
        self._neo4j_password = os.getenv("NEO4J_PASSWORD")
        self._neo4j_driver: Driver | None = None

        self.CHAINLIT_CLIENT_PORT = int(os.getenv("CHAINLIT_CLIENT_PORT", 3546))
        self.AGENT_SERVER_PORT = self.get_agent_server_port()
        self._neo4j_initialized = True

    @property
    def neo4j_driver(self) -> Driver:
        """
        Inizializza in modo lazy e restituisce il driver condiviso di Neo4j.

        Il driver viene creato solo al primo accesso utilizzando le variabili
        d'ambiente di configurazione e viene poi riutilizzato nei successivi
        accessi. Subito dopo la creazione viene eseguito un controllo di
        connettività.

        Raise:
            ValueError: se una o più variabili d'ambiente richieste per Neo4j
                non sono valorizzate.
            Exception: se la creazione del driver o la verifica di connettività
                falliscono.

        Return:
            Driver: l'istanza condivisa del driver Neo4j.
        """
        if self._neo4j_driver is None:
            try:
                if not self._neo4j_uri or not self._neo4j_username or not self._neo4j_password:
                    self.logger.error(
                        "Missing Neo4j environment variables: "
                        "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
                    )
                    raise ValueError(
                        "Missing Neo4j environment variables: "
                        "NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
                    )

                self.logger.info("Connecting to Neo4j...")
                self._neo4j_driver = GraphDatabase.driver(
                    uri=self._neo4j_uri,
                    auth=(self._neo4j_username, self._neo4j_password),
                )
                self._neo4j_driver.verify_connectivity()
                self.logger.success("Neo4j driver created and connection verified")
            except Exception as e:
                self.logger.error(f"Neo4j driver not created correctly: {e}")
                raise

        return self._neo4j_driver

    def _shutdown(self):
        super()._shutdown()

        if self._neo4j_driver is not None:
            self.logger.info("Closing Neo4j driver")
            self._neo4j_driver.close()
            self._neo4j_driver = None


config = Config()
