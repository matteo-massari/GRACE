import os
import logging
from neo4j import Driver, GraphDatabase
from dotenv import load_dotenv


class Config:
    def __init__(self):
        if getattr(self, "_neo4j_initialized", False):
            return

        load_dotenv()

        self._neo4j_uri = os.getenv("NEO4J_URI")
        self._neo4j_username = os.getenv("NEO4J_USERNAME")
        self._neo4j_password = os.getenv("NEO4J_PASSWORD")
        self._neo4j_driver: Driver | None = None

        self._logger = logging.getLogger(__name__)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

        self._neo4j_initialized = True

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def neo4j_driver(self) -> Driver:
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
                self.logger.info("Neo4j driver created and connection verified")
            except Exception as e:
                self.logger.exception(f"Neo4j driver not created correctly: {e}")
                raise

        return self._neo4j_driver

    def _shutdown(self):
        if self._neo4j_driver is not None:
            self.logger.info("Closing Neo4j driver")
            self._neo4j_driver.close()
            self._neo4j_driver = None


config = Config()
