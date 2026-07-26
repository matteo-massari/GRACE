import logging
from typing import Literal

import torch
from torch_geometric import EdgeIndex
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from graph_recc_sys.utils.config import DataClass


class GraphLoader(DataClass):
    """
    Utility class for building temporal train/validation/test splits and
    corresponding PyG loaders for link prediction on a heterogeneous user-item graph.

    The class sorts interactions by time, creates temporal edge splits, builds
    subgraphs containing only the allowed context edges for each phase, and
    instantiates `LinkNeighborLoader` objects for training and validation.

    It also prepares:
    - edge label indices for training, validation, and test
    - exclusion links for validation and test, used to filter out already-seen
      interactions from recommendation candidates

    The split strategy is temporal:
    - training targets come from the earliest interactions
    - validation targets come after training
    - test targets come after validation
    - validation and test context only include training edges

    Args:
        data (HeteroData):
            Input heterogeneous graph containing user-item interactions and timestamps.
        hyperparams:
            Configuration object containing batching and neighborhood sampling
            hyperparameters.

    Attributes:
        data (HeteroData):
            Full heterogeneous graph.
        per_val (float):
            Fraction of edges reserved for validation.
        per_test (float):
            Fraction of edges reserved for test.
        batch_size (int):
            Batch size used by the link neighbor loaders.
        num_neighbors (list[int]):
            Number of sampled neighbors per hop.
        mode (str):
            Negative sampling mode passed to `LinkNeighborLoader`.

        train_loader (LinkNeighborLoader | None):
            Training loader.
        val_loader (LinkNeighborLoader | None):
            Validation loader.

        train_subgraph (HeteroData | None):
            Subgraph used for training.
        val_subgraph (HeteroData | None):
            Subgraph used for validation.
        test_subgraph (HeteroData | None):
            Subgraph used for test-time encoding/evaluation.

        train_edge_label_index (torch.Tensor | None):
            Positive training edge indices used as labels.
        val_edge_label_index (EdgeIndex | None):
            Validation target edge indices.
        test_edge_label_index (EdgeIndex | None):
            Test target edge indices.

        val_exclude_links (EdgeIndex | None):
            Validation edges to exclude from ranking because they were already seen
            in the context.
        test_exclude_links (EdgeIndex | None):
            Test edges to exclude from ranking because they were already seen
            in the context.
    """

    def __init__(self, data: HeteroData, hyperparams):
        """
        Initialize the graph loader.

        Args:
            data (HeteroData):
                Input heterogeneous graph.
            hyperparams:
                Object containing data-loading hyperparameters such as batch size,
                number of sampled neighbors, number of hops, and negative sampling mode.
        """
        super().__init__()
        self.data = data
        self.per_val = 0.2
        self.per_test = 0.2
        self.batch_size = hyperparams.n_batch
        self.num_neighbors = [hyperparams.n_sample_at_node] * hyperparams.n_hop
        self.mode = hyperparams.mode

        # Loaders
        self.train_loader = None
        self.val_loader = None

        # Subgraphs
        self.train_subgraph = None
        self.val_subgraph = None
        self.test_subgraph = None

        # Label indices
        self.train_edge_label_index = None
        self.val_edge_label_index = None
        self.test_edge_label_index = None

        # Excluded links
        self.test_exclude_links = None
        self.val_exclude_links = None

    def _eval_memory(self):
        """
        Estimate the upper bound on the number of sampled connections and nodes.

        This method computes a rough estimate of how many nodes may be involved
        per batch given:
        - the number of hops
        - the number of sampled neighbors per hop
        - the batch size

        The estimate is logged for debugging and memory-awareness purposes.
        """
        connections_per_seed = sum(self.num_neighbors[0] ** i for i in range(len(self.num_neighbors) + 1))
        max_nodes_involved = connections_per_seed * self.batch_size
        logging.info(
            f"Maximum Number of conncetions involved x seed: {connections_per_seed}\n"
            f"max nodes involved: {max_nodes_involved}"
        )

    def _add_n_id(self, subgraph):
        """
        Add global node identifiers to a subgraph.

        This replicates the behavior of PyG neighbor loaders by attaching
        `n_id` tensors containing global node ids for each node type.

        Args:
            subgraph (HeteroData):
                Input subgraph.

        Returns:
            HeteroData:
                The same subgraph with `n_id` fields added for user and item nodes.
        """
        subgraph[self.user_name].n_id = torch.arange(
            self.data[self.user_name].num_nodes,
            dtype=torch.long,
        )

        subgraph[self.item_name].n_id = torch.arange(
            self.data[self.item_name].num_nodes,
            dtype=torch.long,
        )

        return subgraph

    def _build_exclude_links(self, context_index, target_index, sparse_size, label):
        """
        Build target edge labels and exclusion links while checking for leakage.

        This method:
        - creates the `edge_label_index` for the target split
        - builds a set of context edges to exclude from candidate recommendations
          because they were already observed
        - removes any overlap between context and target edges
        - verifies that no target edge leaks into the exclusion set

        The overlap checks are performed using hashed edge ids for efficiency.

        Args:
            context_index (torch.Tensor):
                Indices of context edges used to build the observed interaction set.
            target_index (torch.Tensor):
                Indices of target edges to be predicted/evaluated.
            sparse_size (tuple[int, int]):
                Shape of the user-item adjacency matrix:
                `(num_users, num_items)`.
            label (str):
                Human-readable split label used in logging
                (for example `"val"` or `"test"`).

        Returns:
            tuple[EdgeIndex, EdgeIndex]:
                A tuple containing:
                - `edge_label_index`: sorted target edges
                - `exclude_links`: sorted context edges to exclude at ranking time

        Raises:
            ValueError:
                If any target edge is also found inside the exclusion links,
                indicating data leakage.
        """
        edge_type = (self.user_name, self.user_to_product, self.item_name)
        max_item_id = sparse_size[1]

        edge_label_index = EdgeIndex(
            self.data[edge_type].edge_index[:, target_index],
            sparse_size=sparse_size,
        ).sort_by("row")[0]

        context_edges = self.data[edge_type].edge_index[:, context_index]
        target_edges = self.data[edge_type].edge_index[:, target_index]

        # Remove any target edges from the exclusion candidates
        context_hashes = context_edges[0] * max_item_id + context_edges[1]
        target_hashes = target_edges[0] * max_item_id + target_edges[1]

        mask = ~torch.isin(context_hashes, target_hashes)
        filtered_context_edges = context_edges[:, mask]

        logging.info(
            f"[{label}] Exclude-link filtering summary: \n"
            f"context_edges={context_edges.size(1)},\n"
            f"filtered_exclude_links={filtered_context_edges.size(1)},\n"
            f"removed_overlap_with_target={context_edges.size(1) - filtered_context_edges.size(1)}"
        )

        exclude_links = EdgeIndex(
            filtered_context_edges,
            sparse_size=sparse_size,
        ).sort_by("row")[0]

        # Final leakage check
        edge_label_hashes = edge_label_index[0] * max_item_id + edge_label_index[1]
        exclude_hashes = exclude_links[0] * max_item_id + exclude_links[1]

        overlap_mask = torch.isin(edge_label_hashes, exclude_hashes)
        num_overlap = int(overlap_mask.sum())

        if num_overlap > 0:
            overlapping_edges = edge_label_index[:, overlap_mask]
            logging.error(
                f"[{label}] Data leakage detected: {num_overlap} edges are present in both "
                f"edge_label_index and exclude_links"
            )
            logging.error(f"[{label}] Overlapping edges: {overlapping_edges[:, :10]}")
            raise ValueError(
                f"[{label}] Found {num_overlap} overlapping edges between edge_label_index and exclude_links"
            )
        else:
            logging.info(f"[{label}] Check passed: no overlap between edge_label_index and exclude_links")

        return edge_label_index, exclude_links

    def _split(self):
        """
        Create temporal train/validation/test splits and corresponding loaders.

        The method:
        1. Sorts edges by time
        2. Splits them into train, validation, and test segments
        3. Builds context subgraphs for each split
        4. Adds global node ids to the subgraphs
        5. Creates `LinkNeighborLoader` instances for training and validation
        6. Builds target edge labels and exclusion links for validation and test

        Split policy:
        - train targets: training edges
        - validation targets: validation edges
        - test targets: test edges
        - validation context: only training edges
        - test context: only training edges
        """
        edge_type = (self.user_name, self.user_to_product, self.item_name)
        rev_edge_type = (self.item_name, self.product_to_user, self.user_name)

        perm = self.data[edge_type].time.argsort()

        self.data[edge_type].edge_index = self.data[edge_type].edge_index[:, perm]
        self.data[edge_type].time = self.data[edge_type].time[perm]
        self.data[rev_edge_type].edge_index = self.data[rev_edge_type].edge_index[:, perm]
        self.data[rev_edge_type].time = self.data[rev_edge_type].time[perm]

        time = self.data[edge_type].time
        num_edges = time.numel()

        num_test_edges = int(self.per_test * num_edges)
        num_val_edges = int(self.per_val * num_edges)
        num_train_edges = num_edges - num_val_edges - num_test_edges

        logging.info(f"Split: train={num_train_edges}, val={num_val_edges}, test={num_test_edges}")

        train_target_index = torch.arange(0, num_train_edges)
        val_target_index = torch.arange(num_train_edges, num_train_edges + num_val_edges)
        test_target_index = torch.arange(num_train_edges + num_val_edges, num_edges)

        train_context_index = torch.arange(0, num_train_edges)
        val_context_index = torch.arange(0, num_train_edges)
        test_context_index = torch.arange(0, num_train_edges)

        train_subgraph = self.data.edge_subgraph(
            {edge_type: train_context_index, rev_edge_type: train_context_index}
        )
        self.val_subgraph = self.data.edge_subgraph(
            {edge_type: val_context_index, rev_edge_type: val_context_index}
        )
        self.test_subgraph = self.data.edge_subgraph(
            {edge_type: test_context_index, rev_edge_type: test_context_index}
        )

        # Preserve global node ids inside each subgraph
        self.train_subgraph = self._add_n_id(train_subgraph)
        self.val_subgraph = self._add_n_id(self.val_subgraph)
        self.test_subgraph = self._add_n_id(self.test_subgraph)

        self.train_edge_label_index = train_subgraph[edge_type].edge_index[:, train_target_index]
        val_edge_label_index = self.data[edge_type].edge_index[:, val_target_index]

        self.train_loader = LinkNeighborLoader(
            data=self.train_subgraph,
            num_neighbors=self.num_neighbors,
            edge_label_index=(edge_type, self.train_edge_label_index),
            edge_label_time=time[train_target_index] - 1,
            batch_size=self.batch_size,
            neg_sampling=dict(mode=self.mode, amount=1),
            time_attr="time",
            temporal_strategy="last",
            disjoint=False,
            shuffle=True,
        )

        logging.info("Batching Info: train")
        logging.info(f"batch size, numero di connessioni campionati: {self.batch_size}")
        logging.info(f"Numero di hop: {len(self.num_neighbors)}")
        logging.info(f"test: {next(iter(self.train_loader))}")

        self.val_loader = LinkNeighborLoader(
            data=self.val_subgraph,
            num_neighbors=self.num_neighbors,
            edge_label_index=(edge_type, val_edge_label_index),
            edge_label_time=time[val_target_index] - 1,
            batch_size=self.batch_size,
            neg_sampling=dict(mode=self.mode, amount=1),
            time_attr="time",
            temporal_strategy="last",
            shuffle=False,
        )

        sparse_size = (
            self.data[self.user_name].num_nodes,
            self.data[self.item_name].num_nodes,
        )

        self.test_edge_label_index, self.test_exclude_links = self._build_exclude_links(
            test_context_index, test_target_index, sparse_size, "test"
        )

        self.val_edge_label_index, self.val_exclude_links = self._build_exclude_links(
            train_context_index, val_target_index, sparse_size, "val"
        )

    def get_loaders(self) -> dict:
        """
        Build and return all graph splits, loaders, and evaluation structures.

        This is the main public entry point of the class. It:
        - evaluates an approximate memory footprint
        - creates the temporal split
        - returns all relevant artifacts for training and evaluation

        Returns:
            dict:
                Dictionary containing:
                - `train_subgraph`: training subgraph
                - `train_loader`: training `LinkNeighborLoader`
                - `train_edge_label_index`: training edge labels
                - `val_loader`: validation `LinkNeighborLoader`
                - `val_subgraph`: validation subgraph
                - `test_subgraph`: test subgraph
                - `test_edge_label_index`: test target edges
                - `test_exclude_links`: test exclusion links
                - `val_edge_label_index`: validation target edges
                - `val_exclude_links`: validation exclusion links
        """
        self._eval_memory()
        self._split()
        return {
            "train_subgraph": self.train_subgraph,
            "train_loader": self.train_loader,
            "train_edge_label_index": self.train_edge_label_index,
            "val_loader": self.val_loader,
            "val_subgraph": self.val_subgraph,
            "test_subgraph": self.test_subgraph,
            "test_edge_label_index": self.test_edge_label_index,
            "test_exclude_links": self.test_exclude_links,
            "val_edge_label_index": self.val_edge_label_index,
            "val_exclude_links": self.val_exclude_links,
        }
