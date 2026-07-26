import torch
from torch import device
from torch_geometric import EdgeIndex
from torch_geometric.data import HeteroData
from torch_geometric.metrics import (
    LinkPredAveragePopularity,
    LinkPredHitRatio,
    LinkPredMAP,
    LinkPredNDCG,
    LinkPredPersonalization,
    LinkPredPrecision,
    LinkPredRecall,
)

from graph_recc_sys.utils.config import DataClass


class Evaluation(DataClass):
    """
    Helper class for computing recommendation metrics on user-item predictions.

    This class evaluates top-k recommendation outputs using several families of
    metrics:
    - predictive quality metrics
    - ranking quality metrics
    - behavioral metrics

    The input predictions are expected as a matrix of recommended item indices,
    one row per user, and the ground truth is provided through `edge_label_index`.

    Supported metrics:
    - Precision@K
    - Recall@K
    - MAP@K
    - Hit Ratio@K
    - NDCG@K
    - Average Popularity@K
    - Personalization@K

    Attributes:
        user_name (str):
            Name of the user node type inherited from `DataClass`.
        item_name (str):
            Name of the item node type inherited from `DataClass`.
        user_to_product (str):
            Name of the user-to-item edge relation inherited from `DataClass`.
    """

    def __init__(self) -> None:
        """
        Initialize the evaluation helper.
        """
        super().__init__()

    def _get_popularity(self, subgraph_target: HeteroData) -> torch.Tensor:
        """
        Compute item popularity from a user-item interaction subgraph.

        Popularity is defined as the number of times each item appears in the
        edge list:

        $$
        \text{popularity}(i) = \sum_{(u, j) \in E} \mathbf{1}[j = i]
        $$

        where:
        - $$E$$ is the set of user-item interactions
        - $$i$$ is an item index

        Args:
            subgraph_target (HeteroData):
                Graph whose user-item edges are used to estimate item popularity.

        Returns:
            torch.Tensor:
                Tensor of shape `(num_items,)` containing the interaction count
                for each item.
        """
        edge_index = subgraph_target[(self.user_name, self.user_to_product, self.item_name)].edge_index
        item = edge_index[1]

        num_items = subgraph_target[self.item_name].num_nodes
        popularity = torch.bincount(input=item, minlength=num_items)
        return popularity

    @torch.no_grad()
    def evaluate_epoch(
        self,
        pred_index_mat: torch.Tensor,
        popularity_subgraph: HeteroData,
        edge_label_index: EdgeIndex,
        device: device,
        k: int,
    ) -> dict[str, float]:
        """
        Evaluate top-k predictions against the ground-truth edges.

        The method computes:

        Predictive quality:
        - Precision@K
        - Recall@K

        Ranking quality:
        - MAP@K
        - Hit Ratio@K
        - NDCG@K

        Behavioral metrics:
        - Average Popularity@K
        - Personalization@K

        Args:
            pred_index_mat (torch.Tensor):
                Predicted top-k item indices with shape `(num_users, k)`.
            popularity_subgraph (HeteroData):
                Graph used to estimate item popularity for behavioral metrics.
            edge_label_index (EdgeIndex):
                Ground-truth user-item edges.
            device (device):
                PyTorch device on which the metrics are evaluated.
            k (int):
                Top-k cutoff used for all metrics.

        Returns:
            dict[str, float]:
                Dictionary containing the computed evaluation metrics with keys:
                - `"Precision_at_K"`
                - `"Recall_at_K"`
                - `"MAP_at_K"`
                - `"Hit_at_k"`
                - `"NDCG_at_K"`
                - `"prev_avg_pop_metric"`
                - `"personalization_score"`
        """
        edge_label_index = edge_label_index.to(device)

        # Predictive quality metrics
        precision_metric = LinkPredPrecision(k=k).to(device)
        recall_metric = LinkPredRecall(k=k).to(device)

        # Ranking quality metrics
        map_metric = LinkPredMAP(k=k).to(device)
        hit_metric = LinkPredHitRatio(k=k).to(device)
        ndcg_metric = LinkPredNDCG(k=k).to(device)

        # Behavioral metrics
        popularity = self._get_popularity(subgraph_target=popularity_subgraph)
        prev_avg_pop_metric = LinkPredAveragePopularity(k=k, popularity=popularity).to(device)
        personalization_metric = LinkPredPersonalization(k=k).to(device)

        precision_metric.update(pred_index_mat, edge_label_index)
        recall_metric.update(pred_index_mat, edge_label_index)
        map_metric.update(pred_index_mat, edge_label_index)
        hit_metric.update(pred_index_mat, edge_label_index)
        ndcg_metric.update(pred_index_mat, edge_label_index)
        prev_avg_pop_metric.update(pred_index_mat, edge_label_index)
        personalization_metric.update(pred_index_mat, edge_label_index)

        precision_score = float(precision_metric.compute())
        recall_score = float(recall_metric.compute())
        map_score = float(map_metric.compute())
        hit_score = float(hit_metric.compute())
        ndcg_score = float(ndcg_metric.compute())
        prev_avg_pop_score = float(prev_avg_pop_metric.compute())
        personalization_score = float(personalization_metric.compute())

        return {
            "Precision_at_K": precision_score,
            "Recall_at_K": recall_score,
            "MAP_at_K": map_score,
            "Hit_at_k": hit_score,
            "NDCG_at_K": ndcg_score,
            "prev_avg_pop_metric": prev_avg_pop_score,
            "personalization_score": personalization_score,
        }
