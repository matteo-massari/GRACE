import os
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
import mlflow.pytorch
import yaml
from dotenv import load_dotenv

load_dotenv()


class MlflowConfig:
    """
    Utility class for configuring MLflow authentication, experiment setup,
    and model loading.

    The class reads MLflow-related settings from environment variables and
    provides helper methods to:
    - apply authentication credentials
    - configure the active tracking URI and experiment
    - load a model from a run specified in a YAML configuration file
    - conditionally open an MLflow run through a context manager

    Environment variables used:
    - `MLFLOW_TRACKING_URI`
    - `MLFLOW_EXPERIMENT`
    - `MLFLOW_TRACKING_USERNAME`
    - `MLFLOW_TRACKING_PASSWORDTEST_MLFLOW`

    Attributes:
        tracking_uri (str | None):
            MLflow tracking server URI.
        name_experiment (str | None):
            Name of the MLflow experiment to activate.
        username (str | None):
            Username used for MLflow authentication.
        password (str | None):
            Password used for MLflow authentication.
    """

    def __init__(self) -> None:
        """
        Initialize the MLflow configuration from environment variables.
        """
        self.tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
        self.name_experiment: str | None = os.getenv("MLFLOW_EXPERIMENT")
        self.username: str | None = os.getenv("MLFLOW_TRACKING_USERNAME")
        self.password: str | None = os.getenv("MLFLOW_TRACKING_PASSWORDTEST_MLFLOW")

    @property
    def _apply_auth(self) -> None:
        """
        Apply MLflow authentication credentials to environment variables.

        If username and/or password are defined, they are written into the
        expected MLflow environment variables.
        """
        if self.username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        if self.password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password

    def _get_run(self, config_path: str = "interface.yaml") -> tuple[str, str]:
        """
        Read the target run id and model name from a YAML configuration file.

        The configuration file is expected to contain a structure like:

            settings:
              mlflow:
                run_id: ...
                model_name: ...

        Args:
            config_path (str):
                Path to the YAML configuration file.

        Returns:
            tuple[str, str]:
                A tuple containing:
                - `run_id`
                - `model_name`
        """
        with open(config_path, encoding="utf-8") as f:
            interface = yaml.safe_load(f)
            mlflow_settings = interface["settings"]["mlflow"]
        return mlflow_settings["run_id"], mlflow_settings["model_name"]

    def set_experiment(self) -> None:
        """
        Configure the MLflow tracking server and set the active experiment.

        This method:
        - applies authentication credentials
        - sets the MLflow tracking URI
        - selects the active experiment where runs will be logged
        """
        self._apply_auth
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.name_experiment)

    def get_model(self, device: Any, config_path: str = "interface.yaml") -> Any:
        """
        Load a PyTorch model from MLflow using a YAML configuration file.

        The method reads the target run id and model artifact name from the
        configuration file, then loads the corresponding model from MLflow.

        Args:
            device (Any):
                Device or map location passed to `mlflow.pytorch.load_model`.
            config_path (str):
                Path to the YAML configuration file.

        Returns:
            Any:
                The PyTorch model loaded from MLflow.
        """
        self._apply_auth
        run_id, model_name = self._get_run(config_path)
        model = mlflow.pytorch.load_model(
            model_uri=f"runs:/{run_id}/{model_name}",
            map_location=device,
        )
        return model

    @staticmethod
    @contextmanager
    def mlflow_run(dry_run: bool, run_name: str) -> Iterator[Any | None]:
        """
        Open an MLflow run only when logging is enabled.

        If `dry_run=True`, this context manager yields `None` and does not start
        any MLflow run. Otherwise, it starts a run with the given name and yields
        the active run object.

        Args:
            dry_run (bool):
                If True, disable MLflow logging.
            run_name (str):
                Name of the MLflow run.

        Yields:
            Iterator[Any | None]:
                The active MLflow run object, or None when `dry_run=True`.
        """
        if dry_run:
            yield None
        else:
            with mlflow.start_run(run_name=run_name) as run:
                yield run
