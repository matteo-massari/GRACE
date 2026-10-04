import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

def _external_query(name, *args):
    try:
        from graph_recc_sys import query
    except ImportError as exc:
        raise RuntimeError(
            "CSV locale mancante. Esegui make synthetic-data e imposta DATA_PATH=database/gnn. "
            "Impala extraction requires the private graph_recc_sys.query module."
        ) from exc
    return getattr(query, name)(*args)


load_dotenv()


class DatasetManager:
    """
    Manage dataset extraction and local persistence for training data,
    recommendation tables, and analysis tables.

    Training datasets and recommendation tables are defined consistently from
    the provided hyperparameters, while analysis tables are downloaded through
    a separate flow.

    Expected directory structure::

        DATA_PATH/
        ├── training_tables/
        │   ├── dataset_<n_clients>_<min_connections>_<n_months>/
        │   │   ├── basket_data.csv
        │   │   ├── customer.csv
        │   │   ├── product.csv
        │   │   ├── product_names.csv
        │   │   └── graph_data/
        ├── rec_tables/
        │   └── <k>/
        │       ├── recommendationenginepredict_<k>.csv
        │       ├── bestseller_<k>.csv
        │       └── dataset_res_<k>_features_<add_features>.csv
        └── analysis_tables/
            └── dataset_<n_clients>_<min_connections>_<n_months>/
                ├── basket_data.csv
                ├── customer.csv
                └── product.csv

    Args:
        hyperparams (Any | None):
            Configuration object containing dataset and recommendation settings.
            If provided, the manager initializes all paths needed for training
            and recommendation-table downloads.

    Attributes:
        base_data_path (Path):
            Root directory used to store all generated data. Read from the
            `DATA_PATH` environment variable, defaulting to the current directory.
        training_base_dir (Path):
            Base directory for training datasets.
        rec_base_dir (Path):
            Base directory for recommendation tables.
        analysis_base_dir (Path):
            Base directory for analysis datasets.

        hyperparams (Any):
            Hyperparameter/configuration object passed at initialization.

        n_customers (int | None):
            Number of clients used to define the dataset.
        min_connections (int | None):
            Minimum number of interactions/connections required.
        k (int | None):
            Top-k cutoff used for recommendation tables.
        n_months (int | None):
            Number of months used to filter the dataset.

        dataset_dir (Path):
            Directory containing the training dataset files.
        graph_dir (Path):
            Directory used to store graph-related processed artifacts.
        market_basket_path (Path):
            CSV path for user-product interactions.
        product_desc_path (Path):
            CSV path for product descriptive features.
        product_names_path (Path):
            CSV path for product names.
        customer_desc_path (Path):
            CSV path for customer descriptive features.

        rec_table_dir (Path):
            Directory containing recommendation tables for the selected `k`.
        rec_engine_path (Path):
            CSV path for the recommendation engine output.
        rec_bestseller_path (Path):
            CSV path for the bestseller baseline output.
        rec_gnn_path (Path):
            CSV path intended for GNN recommendation output.

        analysis_dir (Path):
            Directory containing analysis tables for a specific dataset setup.
        analysis_basket_path (Path):
            CSV path for the analysis interaction table.
        analysis_product_path (Path):
            CSV path for the analysis product table.
        analysis_customer_path (Path):
            CSV path for the analysis customer table.
    """

    def __init__(self, hyperparams: Any | None = None) -> None:
        """
        Initialize the dataset manager and compute storage paths.

        If `hyperparams` is provided, training and recommendation-table paths
        are initialized immediately. Otherwise, only the base directories are set.

        Args:
            hyperparams (Any | None):
                Configuration object containing:
                - `n_customers`
                - `min_connections`
                - `k`
                - `n_months`
                - `add_features`
        """
        self.base_data_path: Path = Path(os.getenv("DATA_PATH", "."))

        self.training_base_dir: Path = self.base_data_path / "training_tables"
        self.rec_base_dir: Path = self.base_data_path / "rec_tables"
        self.analysis_base_dir: Path = self.base_data_path / "analysis_tables"

        if hyperparams is not None:
            self.hyperparams: Any = hyperparams
            self.n_customers: int = hyperparams.n_customers
            self.min_connections: int = hyperparams.min_connections
            self.k: int = hyperparams.k
            self.n_months: int = hyperparams.n_months

            self.dataset_dir: Path = (
                self.training_base_dir / f"dataset_{self.n_customers}_{self.min_connections}_{self.n_months}"
            )
            self.graph_dir: Path = self.dataset_dir / "graph_data"
            self.market_basket_path: Path = self.dataset_dir / "basket_data.csv"
            self.product_desc_path: Path = self.dataset_dir / "product.csv"
            self.product_names_path: Path = self.dataset_dir / "product_names.csv"
            self.customer_desc_path: Path = self.dataset_dir / "customer.csv"

            self.rec_table_dir: Path = self.rec_base_dir / str(self.k)
            self.rec_engine_path: Path = self.rec_table_dir / f"recommendationenginepredict_{self.k}.csv"
            self.rec_bestseller_path: Path = self.rec_table_dir / f"bestseller_{self.k}.csv"
            self.rec_gnn_path: Path = (
                self.rec_table_dir / f"dataset_res_{self.k}_features_{self.hyperparams.add_features}.csv"
            )

        else:
            self.n_customers: int | None = None
            self.min_connections: int | None = None
            self.k: int | None = None
            self.n_months: int | None = None

    def _init_analysis_paths(self) -> None:
        """
        Initialize the paths used to store analysis tables.

        This method uses the current values of:
        - `self.n_customers`
        - `self.min_connections`
        - `self.n_months`

        and builds the corresponding analysis directory structure.
        """
        self.analysis_dir: Path = (
            self.analysis_base_dir / f"dataset_{self.n_customers}_{self.min_connections}_{self.n_months}"
        )
        self.analysis_basket_path: Path = self.analysis_dir / "basket_data.csv"
        self.analysis_product_path: Path = self.analysis_dir / "product.csv"
        self.analysis_customer_path: Path = self.analysis_dir / "customer.csv"

    def _fetch(self, query: str, output_path: Path) -> None:
        """
        Execute a query through Impala and save the result as a CSV file.

        The parent directory is created automatically if it does not exist.

        Args:
            query (str):
                SQL query to execute.
            output_path (Path):
                Destination path where the resulting CSV will be written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from impala.util import as_pandas
        from maisutils.impala_utils import close_impyla_conn, open_impyla_conn

        connection, cursor = open_impyla_conn()
        try:
            cursor.execute(query)
            df = as_pandas(cursor)
            df.to_csv(output_path, index=False)

            logging.info("File saved successfully: %s", output_path)
        finally:
            close_impyla_conn(connection, cursor)

    def build(self) -> str:
        """
        Create the dataset required for model training and inference.

        This method downloads and stores, if missing:
        - market basket interactions
        - product features
        - customer features
        - product names

        Returns:
            str:
                Path to the dataset directory as a string.
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        if not self.market_basket_path.exists():
            self._fetch(
                _external_query("_query_market_basket", self.n_customers, self.min_connections, self.n_months),
                self.market_basket_path,
            )

        if not self.product_desc_path.exists():
            self._fetch(
                _external_query("_query_product", self.n_customers, self.min_connections, self.n_months),
                self.product_desc_path,
            )

        if not self.customer_desc_path.exists():
            self._fetch(
                _external_query("_query_customer", self.n_customers, self.min_connections, self.n_months),
                self.customer_desc_path,
            )

        if not self.product_names_path.exists():
            self._fetch(_external_query("_query_product_names", ), self.product_names_path)

        return str(self.dataset_dir)

    def download_rec_table(self) -> tuple[str, str]:
        """
        Download the recommendation tables for the configured top-k value.

        This method requires `hyperparams` to have been provided at construction
        time, since `k` and related paths are derived from it.

        It downloads, if missing:
        - the recommendation engine output
        - the bestseller baseline output

        Returns:
            tuple[str, str]:
                A tuple containing:
                - the recommendation engine CSV path
                - the bestseller CSV path

        Raises:
            ValueError:
                If the manager was initialized without `hyperparams`.
        """
        if self.k is None:
            raise ValueError("hyperparams required for download_rec_table()")

        self.rec_table_dir.mkdir(parents=True, exist_ok=True)

        if not self.rec_engine_path.exists():
            self._fetch(_external_query("_query_rec_engine", self.k), self.rec_engine_path)

        if not self.rec_bestseller_path.exists():
            self._fetch(_external_query("_query_bestseller", self.k), self.rec_bestseller_path)

        return str(self.rec_engine_path), str(self.rec_bestseller_path)

    def download_analysis_table(
        self,
        n_customers: int,
        min_connections: int,
        n_months: int,
    ) -> str:
        """
        Download the enriched tables used for analysis.

        The tables are stored under:

            analysis_tables/<dataset_name>/

        For this call, the provided parameters temporarily overwrite the current
        dataset configuration stored in the instance.

        Args:
            n_customers (int):
                Number of clients to include.
            min_connections (int):
                Minimum number of interactions/connections required.
            n_months (int):
                Number of months to include.

        Returns:
            str:
                Path to the analysis directory as a string.
        """
        self.n_customers = n_customers
        self.min_connections = min_connections
        self.n_months = n_months

        self._init_analysis_paths()
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        if not self.analysis_basket_path.exists():
            self._fetch(
                _external_query("_query_market_basket", self.n_customers, self.min_connections, self.n_months),
                self.analysis_basket_path,
            )

        if not self.analysis_product_path.exists():
            self._fetch(
                _external_query("_query_product_analysis", self.n_customers, self.min_connections, self.n_months),
                self.analysis_product_path,
            )

        if not self.analysis_customer_path.exists():
            self._fetch(
                _external_query("_query_customer_analysis", self.n_customers, self.min_connections, self.n_months),
                self.analysis_customer_path,
            )

        return str(self.analysis_dir)
