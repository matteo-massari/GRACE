import logging

import torch
import torch.nn.functional as F
from torch import device
from torch_geometric import EdgeIndex
from torch_geometric.data import HeteroData

from graph_recc_sys.utils.config import DataClass


class Inference(DataClass):
    """
    Helper class for model inference and top-k recommendation generation.

    This class computes user-to-item recommendation candidates starting from
    node embeddings produced by the model encoder. It supports similarity-based
    ranking through matrix multiplication and can exclude already-observed
    user-item edges from the candidate set.

    The main workflow is:
    1. encode the heterogeneous graph with the model
    2. extract user and item embeddings
    3. compute similarity scores between all users and all items
    4. mask excluded edges
    5. return the top-k item indices for each user

    Attributes:
        user_name (str):
            Name of the user node type inherited from `DataClass`.
        item_name (str):
            Name of the item node type inherited from `DataClass`.
    """

    def __init__(self) -> None:
        """
        Initialize the inference helper.
        """
        super().__init__()

    @staticmethod
    def mips_search(
        query_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        k: int,
        similarity: str,
        exclude_links: EdgeIndex | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform top-k retrieval over destination embeddings for each query
        embedding using a similarity score matrix.

        If `similarity == "cosine"`, both query and destination embeddings are
        L2-normalized before scoring. In that case, matrix multiplication is
        equivalent to cosine similarity.

        Otherwise, the method uses raw dot-product scores:

        $$
        S = Q D^T
        $$

        where:
        - $$Q$$ is the query embedding matrix
        - $$D$$ is the destination embedding matrix

        Excluded edges are masked by setting their score to $$-\infty$$ before
        top-k extraction.

        Args:
            query_emb (torch.Tensor):
                Query embedding matrix of shape `(num_queries, embedding_dim)`.
            dst_emb (torch.Tensor):
                Destination embedding matrix of shape `(num_items, embedding_dim)`.
            k (int):
                Number of top items to retrieve for each query.
            similarity (str):
                Similarity type. If `"cosine"`, embeddings are normalized before
                scoring. Otherwise, raw dot product is used.
            exclude_links (EdgeIndex | torch.Tensor | None):
                Optional tensor of shape `(2, num_excluded_edges)` containing
                `(row, col)` pairs to mask from the score matrix.

        Returns:
            torch.Tensor:
                Tensor of top-k predicted item indices with shape
                `(num_queries, k)`.
        """
        if similarity == "cosine":
            query_emb = F.normalize(query_emb, dim=-1)
            dst_emb = F.normalize(dst_emb, dim=-1)

        scores = query_emb @ dst_emb.T

        if exclude_links is not None:
            rows = exclude_links[0]
            cols = exclude_links[1]

            mask = (rows >= 0) & (rows < scores.size(0)) & (cols >= 0) & (cols < scores.size(1))

            valid_rows = rows[mask]
            valid_cols = cols[mask]

            if valid_rows.numel() > 0:
                scores[valid_rows, valid_cols] = float("-inf")
        else:
            logging.warning("No exclude_links were provided. This may cause leakage.")

        _, pred_index_mat = scores.topk(k, dim=-1)
        return pred_index_mat

    @torch.no_grad()
    def test_epoch_dist(
        self,
        model: torch.nn.Module,
        subgraph: HeteroData,
        exclude_links: EdgeIndex,
        device: device,
        similarity: str,
        k: int,
    ) -> torch.Tensor:
        """
        Run full-graph inference on a subgraph and return top-k item
        recommendations for each user.

        The method:
        1. moves the subgraph and exclusion links to the target device
        2. encodes the graph with the model
        3. extracts user and item embeddings
        4. performs top-k retrieval with exclusion masking

        Args:
            model (torch.nn.Module):
                Trained recommendation model. It must expose an `encoder()`
                method returning a dictionary of node embeddings.
            subgraph (HeteroData):
                Graph used for inference.
            exclude_links (EdgeIndex):
                User-item edges to exclude from the recommendation candidates.
            device (device):
                PyTorch device used for inference.
            similarity (str):
                Similarity metric used for ranking.
            k (int):
                Number of top recommendations to return per user.

        Returns:
            torch.Tensor:
                Tensor of predicted item indices with shape `(num_users, k)`.
        """
        model.eval()

        subgraph = subgraph.to(device)
        exclude_links = exclude_links.to(device)

        emb_dict = model.encoder(subgraph)

        src_emb = emb_dict[self.user_name]
        dst_emb = emb_dict[self.item_name]

        pred_index_mat = self.mips_search(
            query_emb=src_emb,
            dst_emb=dst_emb,
            similarity=similarity,
            k=k,
            exclude_links=exclude_links,
        )
        return pred_index_mat
