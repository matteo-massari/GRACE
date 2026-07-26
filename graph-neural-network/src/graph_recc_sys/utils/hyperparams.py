from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass
class HyperParams:
    """
    Container for all hyperparameters used across the recommendation pipeline.

    This dataclass groups together configuration values related to:
    - optimization and training
    - model architecture
    - graph construction and sampling
    - evaluation
    - hardware selection

    The expected values are typically loaded from a YAML configuration file
    through the `from_yaml()` class method.

    Attributes:
        lr (float):
            Initial learning rate.
        eta_min (float):
            Minimum learning rate used by the cosine annealing scheduler.
        warmup_period (int):
            Number of warmup steps (or warmup multiplier, depending on usage)
            before the main scheduler takes over.
        n_epochs (int):
            Number of training epochs.
        margin (int):
            Margin used by margin-based losses such as triplet loss.

        n_conv_layer (int):
            Number of graph convolution layers.
        n_projs_layer (int | None):
            Number of hidden projection layers before the final output layer.
        conv_type (Literal["GraphSage"]):
            Type of graph convolution used in the model.
        loss_function (Literal["TripletMarginLoss", "BPRLoss", "InfoNCE", "BCEWithLogitsLoss", "FocalLoss"]):
            Training loss function.
        input_channels (int):
            Dimension of the input node representation.
        hidden_channels (int):
            Hidden dimension used inside the model.
        out_channels (int):
            Final embedding dimension.
        dropout_rate (float):
            Dropout probability.
        linear_init (bool):
            Whether to apply custom initialization to linear layers.
        conv_init (bool):
            Whether to apply custom initialization to convolution layers.
        two_tower (bool):
            Whether to use separate projection towers for users and items.
        proj_norm (bool):
            Whether projection normalization is enabled.
        add_features (bool):
            Whether node features are used as model input.
        num_stochastic_item_features (int):
            Number of additional learnable stochastic item features.
        num_stochastic_user_features (int):
            Number of additional learnable stochastic user features.
        aggr (Literal["mean", "max"]):
            Aggregation strategy used in graph convolutions.

        n_clienti (int | None):
            Number of customers used for dataset extraction.
        n_mesi (int):
            Number of months used to define the data window.
        n_connessioni_minime (int):
            Minimum number of interactions required per customer/entity.

        n_batch (int):
            Batch size used by the graph loader.
        n_hop (int):
            Number of neighborhood expansion hops.
        n_sample_at_node (int):
            Number of sampled neighbors per node and per hop.

        mode (Literal["binary", "triplet"]):
            Training/inference mode of the model.
        similarity (Literal["Euclidean", "Cosine"]):
            Similarity or distance metric used at inference time.
        k (int):
            Top-k cutoff used for recommendation generation and evaluation.
        gpu (int):
            GPU index to use.
    """

    lr: float
    eta_min: float
    warmup_period: int
    n_epochs: int
    margin: int
    n_conv_layer: int
    n_projs_layer: int | None
    conv_type: Literal["GraphSage"]
    loss_function: Literal["TripletMarginLoss", "BPRLoss", "InfoNCE", "BCEWithLogitsLoss", "FocalLoss"]
    input_channels: int
    hidden_channels: int
    out_channels: int
    dropout_rate: float
    linear_init: bool
    conv_init: bool
    two_tower: bool
    proj_norm: bool
    add_features: bool
    num_stochastic_item_features: int
    num_stochastic_user_features: int
    aggr: Literal["mean", "max"]
    n_clienti: int | None
    n_mesi: int
    n_connessioni_minime: int
    n_batch: int
    n_hop: int
    n_sample_at_node: int
    mode: Literal["binary", "triplet"]
    similarity: Literal["Euclidean", "Cosine"]
    k: int
    gpu: int

    @classmethod
    def from_yaml(cls, config_path: str = "interface.yaml") -> "HyperParams":
        """
        Build a `HyperParams` instance from a YAML configuration file.

        The YAML file is expected to contain a top-level `hyperparameters`
        section with the following nested blocks:
        - `training`
        - `model`
        - `graph`
        - `loader`
        - `eval`

        These blocks are merged and passed to the dataclass constructor.

        Args:
            config_path (str):
                Path to the YAML configuration file.

        Returns:
            HyperParams:
                Instantiated hyperparameter object.
        """
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)["hyperparameters"]

        return cls(
            **data["training"],
            **data["model"],
            **data["graph"],
            **data["loader"],
            **data["eval"],
        )
