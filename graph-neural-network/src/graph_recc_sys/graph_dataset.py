import hashlib
import os
import pickle
from typing import Any

import pandas as pd
import torch
from torch_geometric import transforms as T
from torch_geometric.data import HeteroData, InMemoryDataset

from graph_recc_sys.utils.config import DataClass
from graph_recc_sys.extraction import DatasetManager


class MarketBasketGraphBuilder(DataClass, InMemoryDataset):
    """
    Build and manage a PyTorch Geometric heterogeneous graph from market basket,
    product master data, and customer master data.

    This class:
    - reads the input CSV files
    - preprocesses the source tables
    - builds a heterogeneous user-item graph
    - attaches node features for users and products
    - saves the processed graph to disk
    - saves and reloads the mappings between original ids and internal indices

    The resulting graph contains:
    - user nodes
    - product nodes
    - edges between users and products representing purchase interactions
    - edge timestamps for possible temporal splits
    - numerical node features for both users and products

    Args:
        dataset_manager (DatasetManager):
            Object containing the dataset paths and the target dataset directory.

    Attributes:
        market_basket_dir (str):
            Path to the CSV file containing user-product interactions.
        product_desc_dir (str):
            Path to the CSV file containing product descriptive features.
        customer_desc_dir (str):
            Path to the CSV file containing customer descriptive features.

        _market_basket (pd.DataFrame | None):
            Raw interaction table loaded from CSV.
        _product_desc (pd.DataFrame | None):
            Raw product feature table loaded from CSV.
        _customer_desc (pd.DataFrame | None):
            Raw customer feature table loaded from CSV.

        user_map (dict[Any, int] | None):
            Mapping from original user identifier to internal graph index.
        item_map (dict[Any, int] | None):
            Mapping from original product identifier to internal graph index.
        user_inv_map (dict[int, Any] | None):
            Reverse mapping from internal user index to original identifier.
        item_inv_map (dict[int, Any] | None):
            Reverse mapping from internal product index to original identifier.

        _hetero_data (HeteroData | None):
            Temporary graph object used during processing.
            The final graph is then stored on disk and reloaded through
            `InMemoryDataset`.
    """

    def __init__(self, dataset_manager: DatasetManager) -> None:
        """
        Initialize the graph builder and load the processed graph from disk.

        If the processed files do not exist yet, PyG will automatically trigger
        `process()` through the `InMemoryDataset` lifecycle.

        Args:
            dataset_manager (DatasetManager):
                Object containing input file paths and the dataset output directory.
        """
        DataClass.__init__(self)

        self.market_basket_dir: str = dataset_manager.market_basket_path
        self.product_desc_dir: str = dataset_manager.product_desc_path
        self.customer_desc_dir: str = dataset_manager.customer_desc_path

        self._market_basket: pd.DataFrame | None = None
        self._product_desc: pd.DataFrame | None = None
        self._customer_desc: pd.DataFrame | None = None

        # Mapping caches
        self.user_map: dict[Any, int] | None = None
        self.item_map: dict[Any, int] | None = None
        self.user_inv_map: dict[int, Any] | None = None
        self.item_inv_map: dict[int, Any] | None = None

        # Temporary graph used during process()
        # Note: the actual persisted graph is stored in self.data by InMemoryDataset
        self._hetero_data: HeteroData | None = None

        # Rebuild the cached graph when any input CSV changes.
        digest = hashlib.sha256()
        for path in (self.market_basket_dir, self.product_desc_dir, self.customer_desc_dir):
            digest.update(path.read_bytes())
        self._source_fingerprint = digest.hexdigest()
        fingerprint_path = dataset_manager.dataset_dir / "processed" / "source.sha256"
        rebuild = (not fingerprint_path.exists()
                   or fingerprint_path.read_text() != self._source_fingerprint)

        InMemoryDataset.__init__(
            self,
            dataset_manager.dataset_dir,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            force_reload=rebuild,
        )

        # Load processed graph from disk
        self.load(self.processed_paths[0])
        self._load_mappings()
        fingerprint_path.write_text(self._source_fingerprint)

    def _read_csv(self) -> None:
        """
        Read the input CSV files into pandas DataFrames.
        """
        self._market_basket = pd.read_csv(self.market_basket_dir)
        self._product_desc = pd.read_csv(self.product_desc_dir)
        self._customer_desc = pd.read_csv(self.customer_desc_dir)

    @property
    def raw_file_names(self) -> list[str]:
        """
        Return the list of raw input files used by the dataset.

        Returns:
            list[str]:
                Paths to the interaction, product, and customer CSV files.
        """
        return [
            self.market_basket_dir,
            self.product_desc_dir,
            self.customer_desc_dir,
        ]

    @property
    def processed_file_names(self) -> list[str]:
        """
        Return the list of processed dataset file names.

        Returns:
            list[str]:
                Names of files generated during processing.
        """
        return ["data.pt"]

    def process(self) -> None:
        """
        Build the heterogeneous graph and save it to disk.

        The processing pipeline:
        1. Read the CSV files
        2. Preprocess the input tables
        3. Build the user-item interaction graph
        4. Add product node features
        5. Add customer node features
        6. Convert the graph to an undirected version
        7. Optionally apply `pre_filter` and `pre_transform`
        8. Save the processed graph
        9. Save the id mappings to disk
        """
        self._read_csv()
        self._preprocessing()
        self._build_graph()
        self._add_product_features()
        self._add_customer_features()

        data = T.ToUndirected()(self._hetero_data)

        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

        with open(os.path.join(self.processed_dir, "mappings.pkl"), "wb") as f:
            pickle.dump(
                {
                    "user_map": self.user_map,
                    "item_map": self.item_map,
                    "user_inv_map": self.user_inv_map,
                    "item_inv_map": self.item_inv_map,
                },
                f,
            )

    def _load_mappings(self) -> None:
        """
        Load user/item index mappings from disk.
        """
        path = os.path.join(self.processed_dir, "mappings.pkl")
        with open(path, "rb") as f:
            maps = pickle.load(f)

        self.user_map = maps["user_map"]
        self.item_map = maps["item_map"]
        self.user_inv_map = maps["user_inv_map"]
        self.item_inv_map = maps["item_inv_map"]

    def _preprocessing(self) -> None:
        """
        Preprocess the raw input tables.

        This method:
        - converts the interaction date column to datetime
        - sorts interactions by date
        - removes duplicate user-item interactions
        - removes duplicate product rows
        - removes duplicate customer rows
        """
        assert self._market_basket is not None
        assert self._product_desc is not None
        assert self._customer_desc is not None

        self._market_basket["dat"] = pd.to_datetime(self._market_basket["dat"], errors="coerce")
        self._market_basket = self._market_basket.sort_values("dat").reset_index(drop=True)

        self._market_basket = self._market_basket.drop_duplicates(subset=[self.user_name, self.item_name])

        self._product_desc = self._product_desc.drop_duplicates(subset=[self.item_name]).copy()
        self._customer_desc = self._customer_desc.drop_duplicates(subset=[self.user_name]).copy()

    def _build_graph(self) -> None:
        """
        Build the base heterogeneous user-item graph.

        This method:
        - extracts unique users and items from the interaction table
        - creates internal index mappings
        - builds the edge index for user-item interactions
        - attaches edge timestamps for temporal modeling
        """
        assert self._market_basket is not None

        market_basket = self._market_basket

        users = market_basket[self.user_name].unique()
        items = market_basket[self.item_name].unique()

        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: i for i, it in enumerate(items)}

        self.user_inv_map = {i: u for u, i in self.user_map.items()}
        self.item_inv_map = {i: it for it, i in self.item_map.items()}

        user_idx = torch.tensor(
            market_basket[self.user_name].map(self.user_map).values,
            dtype=torch.long,
        )
        item_idx = torch.tensor(
            market_basket[self.item_name].map(self.item_map).values,
            dtype=torch.long,
        )

        data = HeteroData()
        data[self.user_name].num_nodes = len(users)
        data[self.item_name].num_nodes = len(items)

        data[self.user_name, self.user_to_product, self.item_name].edge_index = torch.stack(
            [user_idx, item_idx], dim=0
        )

        edge_time = torch.tensor(
            self._market_basket["dat"].to_numpy(dtype="datetime64[s]").astype("int64"),
            dtype=torch.long,
        )

        data[self.user_name, self.user_to_product, self.item_name].time = edge_time
        self._hetero_data = data

    def _add_product_features(self) -> None:
        """
        Add product node features to the graph.

        Feature handling:
        - categorical product columns are factorized into integer ids
        - numerical columns are coerced to float
        - missing values are filled with zero

        The resulting feature matrix is stored in:
            `self._hetero_data[self.item_name].x`

        A dictionary of categorical cardinalities is also stored in:
            `self._hetero_data[self.item_name].cardinalities`
        """
        assert self._product_desc is not None
        assert self.item_map is not None
        assert self._hetero_data is not None

        product_desc = self._product_desc

        feature_cols = list(dict.fromkeys(col for col in self.product_feature_cols if col != self.item_name))

        cardinalities: dict[str, int] = {}
        for col in feature_cols:
            if col in self.product_categorical_cols:
                temp = pd.factorize(product_desc[col])[0]
                temp = temp + 1
                product_desc[col] = temp
                cardinalities[col] = int(temp.max()) + 1

        product_df = pd.DataFrame(
            {
                self.item_name: list(self.item_map.keys()),
                "item_idx": list(self.item_map.values()),
            }
        )

        product_df = product_df.merge(
            product_desc[[self.item_name] + feature_cols],
            on=self.item_name,
            how="left",
        ).sort_values("item_idx")

        product_df[feature_cols] = (
            product_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        )

        self._hetero_data[self.item_name].x = torch.tensor(
            product_df[feature_cols].to_numpy(),
            dtype=torch.float,
        )
        self._hetero_data[self.item_name].cardinalities = cardinalities

    def _add_customer_features(self) -> None:
        """
        Add customer node features to the graph.

        Feature handling:
        - categorical customer columns are factorized into integer ids
        - numerical columns are coerced to float
        - missing values are filled with zero

        The resulting feature matrix is stored in:
            `self._hetero_data[self.user_name].x`

        A dictionary of categorical cardinalities is also stored in:
            `self._hetero_data[self.user_name].cardinalities`
        """
        assert self._customer_desc is not None
        assert self.user_map is not None
        assert self._hetero_data is not None

        customer_desc = self._customer_desc

        feature_cols = list(dict.fromkeys(col for col in self.customer_feature_cols if col != self.user_name))

        cardinalities: dict[str, int] = {}
        for col in feature_cols:
            if col in self.customer_categorical_cols:
                temp = pd.factorize(customer_desc[col])[0]
                temp = temp + 1
                customer_desc[col] = temp
                cardinalities[col] = int(temp.max()) + 1

        customer_df = pd.DataFrame(
            {
                self.user_name: list(self.user_map.keys()),
                "user_idx": list(self.user_map.values()),
            }
        )

        customer_df = customer_df.merge(
            customer_desc[[self.user_name] + feature_cols],
            on=self.user_name,
            how="left",
        ).sort_values("user_idx")

        customer_df[feature_cols] = (
            customer_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        )

        self._hetero_data[self.user_name].x = torch.tensor(
            customer_df[feature_cols].to_numpy(),
            dtype=torch.float,
        )
        self._hetero_data[self.user_name].cardinalities = cardinalities

    @property
    def graph(self) -> HeteroData:
        """
        Return the processed heterogeneous graph.

        Returns:
            HeteroData:
                The processed graph stored by the dataset.
        """
        return self[0]
