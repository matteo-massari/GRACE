import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, Module, ModuleDict, ModuleList, ReLU, Sequential
from torch.nn import LayerNorm as SequentialLayerNorm
from torch.nn import BatchNorm1d as SequentialBatchNorm
from torch_geometric.data import HeteroData
from torch_geometric.nn import BatchNorm as GraphBatchNorm
from torch_geometric.nn import HeteroConv, SAGEConv

from graph_recc_sys.utils.config import DataClass


class RecommenderModel(Module, DataClass):
    """
    Neural model for link prediction on a heterogeneous user-item graph.

    The model builds initial user and item node representations either from
    learnable free embeddings or from available node features through
    embedding/linear transformations. These representations are then propagated
    through a stack of heterogeneous GraphSAGE convolution layers and finally
    projected into a latent output space.

    In `binary` mode, the decoder produces a score for each candidate edge
    using the dot product between user and item embeddings.
    In `triplet` mode, the model directly returns normalized node embeddings,
    which can be used with metric-learning or contrastive objectives.

    Args:
        data (HeteroData):
            Heterogeneous graph containing nodes, node features, and edges.
        hyperparams:
            Configuration object containing all model hyperparameters.

    Hyperparameters:
        n_conv_layer (int):
            Number of heterogeneous convolution layers.
        n_projs_layer (int):
            Number of hidden projection layers before the final output layer.
        dropout_rate (float):
            Dropout rate applied after convolutions and projection layers.
        input_channels (int):
            Dimension of the initial node representations.
        hidden_channels (int):
            Dimension of the hidden representations after graph convolutions.
        out_channels (int):
            Final latent embedding dimension.
        mode (str):
            Output mode of the model. Supported values: `"binary"` or `"triplet"`.
        two_tower (bool):
            If True, uses separate projection heads for users and items.
            Otherwise, a shared projection head is used.
        proj_norm (str | None):
            Normalization type used in projection layers.
            Supported values: `"LayerNorm"`, `"BatchNorm"`, or None.
        aggr (str):
            Aggregation strategy used inside SAGEConv.
        add_features (bool):
            If True, initial node representations are built from node features.
            If False, free embeddings are used for users and items.
        num_stochastic_item_features (int):
            Number of additional free embedding components concatenated to item
            representations when `add_features=True`.
        num_stochastic_user_features (int):
            Number of additional free embedding components concatenated to user
            representations when `add_features=True`.

    Attributes:
        user_emb (Embedding):
            Free user embedding table used when `add_features=False`.
        item_emb (Embedding):
            Free item embedding table used when `add_features=False`.
        user_embeddings (ModuleDict):
            Dictionary of modules used to encode user features when
            `add_features=True`.
        item_embeddings (ModuleDict):
            Dictionary of modules used to encode item features when
            `add_features=True`.
        user_stochastic_emb (Embedding):
            Optional additional learnable embedding for users.
        item_stochastic_emb (Embedding):
            Optional additional learnable embedding for items.
        convs (ModuleList):
            Stack of `HeteroConv` layers based on `SAGEConv`.
        proj_layer (Sequential):
            Shared projection stack used when `two_tower=False`.
        proj_user_layer (Sequential):
            User-specific projection stack used when `two_tower=True`.
        proj_item_layer (Sequential):
            Item-specific projection stack used when `two_tower=True`.
    """

    def __init__(self, data: HeteroData, hyperparams):
        """
        Initialize the recommender model.

        Args:
            data (HeteroData):
                Input heterogeneous graph.
            hyperparams:
                Object containing model hyperparameters.
        """
        Module.__init__(self)
        DataClass.__init__(self)

        # Model hyperparameters
        self.n_conv_layer = hyperparams.n_conv_layer
        self.n_projs_layer = hyperparams.n_projs_layer
        self.dropout_rate = hyperparams.dropout_rate
        self.input_channels = hyperparams.input_channels
        self.hidden_channels = hyperparams.hidden_channels
        self.out_channels = hyperparams.out_channels
        self.mode = hyperparams.mode
        self.two_tower = hyperparams.two_tower
        self.proj_norm = hyperparams.proj_norm
        self.aggr = hyperparams.aggr
        self.add_features = hyperparams.add_features
        self.num_stochastic_item_features = hyperparams.num_stochastic_item_features
        self.num_stochastic_user_features = hyperparams.num_stochastic_user_features

        # Node cardinalities by type
        self.num_users = data[self.user_name].num_nodes
        self.num_items = data[self.item_name].num_nodes

        self.num_users_features = int(data[self.user_name].x.shape[1])
        self.num_items_features = int(data[self.item_name].x.shape[1])

        # Build model blocks
        if self.add_features:
            self._build_embeddings_with_feat(data)
        elif self.add_features is False:
            self._build_embeddings()
        else:
            raise ValueError("True or False")

        # Build convolution and projection layers
        self._build_conv_sage_layer()
        self._build_proj_layer(proj_norm=self.proj_norm)

    def _build_embeddings(self):
        """
        Create free embedding tables for users and items.

        The resulting embeddings have shape:
            - users: `(num_users, input_channels)`
            - items: `(num_items, input_channels)`
        """
        self.user_emb = Embedding(self.num_users, self.input_channels)
        self.item_emb = Embedding(self.num_items, self.input_channels)

    def _build_embeddings_with_feat(self, data: HeteroData):
        """
        Build feature-based encoders for user and item nodes.

        Categorical features are encoded with embedding layers, while
        numerical features are projected with linear layers. Optionally,
        additional learnable stochastic embeddings can be concatenated
        to the resulting representations.

        Args:
            data (HeteroData):
                Input heterogeneous graph containing feature cardinalities.
        """
        self.item_cardinalities = data[self.item_name].cardinalities
        self.user_cardinalities = data[self.user_name].cardinalities

        # Feature columns
        self.item_feature_cols = [col for col in self.product_feature_cols if col != self.item_name]
        self.user_feature_cols = [col for col in self.customer_feature_cols if col != self.user_name]

        # Embedding size per feature so that concatenation approximately matches input_channels
        if len(self.item_feature_cols) >= 1:
            self.item_emb_size = self.input_channels // (
                len(self.item_feature_cols) + self.num_stochastic_item_features
            )
        else:
            self.item_emb_size = self.input_channels

        if len(self.user_feature_cols) >= 1:
            self.user_emb_size = self.input_channels // (
                len(self.user_feature_cols) + self.num_stochastic_user_features
            )
        else:
            self.user_emb_size = self.input_channels

        # Item feature encoders
        self.item_embeddings = ModuleDict()
        for col in self.item_feature_cols:
            if col in self.item_cardinalities:
                self.item_embeddings[col] = Embedding(self.item_cardinalities[col], self.item_emb_size)
            else:
                self.item_embeddings[col] = Linear(1, self.item_emb_size)

        # User feature encoders
        self.user_embeddings = ModuleDict()
        for col in self.user_feature_cols:
            if col in self.user_cardinalities:
                self.user_embeddings[col] = Embedding(self.user_cardinalities[col], self.user_emb_size)
            else:
                self.user_embeddings[col] = Linear(1, self.user_emb_size)

        # Optional stochastic node embeddings
        if self.num_stochastic_item_features > 0:
            self.item_stochastic_emb = Embedding(
                self.num_items, self.item_emb_size * self.num_stochastic_item_features
            )
        if self.num_stochastic_user_features > 0:
            self.user_stochastic_emb = Embedding(
                self.num_users, self.user_emb_size * self.num_stochastic_user_features
            )

    def _build_conv_sage_layer(self):
        """
        Create a stack of heterogeneous GraphSAGE convolution layers.

        Each layer applies message passing in both directions:
            - user -> item
            - item -> user

        The first layer takes `input_channels` as input, while the following
        layers operate on `hidden_channels`.
        """
        self.convs = ModuleList()

        for idx in range(self.n_conv_layer):
            in_channels = self.input_channels if idx == 0 else self.hidden_channels

            self.convs.append(
                HeteroConv(
                    {
                        (self.user_name, self.user_to_product, self.item_name): SAGEConv(
                            (in_channels, in_channels),
                            self.hidden_channels,
                            aggr=self.aggr,
                            normalize=True,
                            project=True,
                        ),
                        (self.item_name, self.product_to_user, self.user_name): SAGEConv(
                            (in_channels, in_channels),
                            self.hidden_channels,
                            aggr=self.aggr,
                            normalize=True,
                            project=True,
                        ),
                    }
                )
            )

    def _build_proj_layer(self, proj_norm):
        """
        Create the projection stack into the final latent space.

        The structure is typically:
            Linear -> Normalization -> ReLU -> Dropout

        followed by a final output projection to `out_channels`.

        Args:
            proj_norm (str | None):
                Type of normalization to use in hidden projection layers.
                Supported values are `"LayerNorm"`, `"BatchNorm"`, or None.
        """
        if self.two_tower:
            self.proj_item_layer = Sequential()
            self.proj_user_layer = Sequential()

            if self.n_projs_layer is not None and proj_norm == "LayerNorm":
                for _ in range(self.n_projs_layer):
                    self.proj_item_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_item_layer.append(SequentialLayerNorm(self.hidden_channels))
                    self.proj_item_layer.append(ReLU())
                    self.proj_item_layer.append(torch.nn.Dropout(self.dropout_rate))

                    self.proj_user_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_user_layer.append(SequentialLayerNorm(self.hidden_channels))
                    self.proj_user_layer.append(ReLU())
                    self.proj_user_layer.append(torch.nn.Dropout(self.dropout_rate))

            if self.n_projs_layer is not None and proj_norm == "BatchNorm":
                for _ in range(self.n_projs_layer):
                    self.proj_item_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_item_layer.append(SequentialBatchNorm(self.hidden_channels))
                    self.proj_item_layer.append(ReLU())
                    self.proj_item_layer.append(torch.nn.Dropout(self.dropout_rate))

                    self.proj_user_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_user_layer.append(SequentialBatchNorm(self.hidden_channels))
                    self.proj_user_layer.append(ReLU())
                    self.proj_user_layer.append(torch.nn.Dropout(self.dropout_rate))

            if self.n_projs_layer is not None and proj_norm is None:
                for _ in range(self.n_projs_layer):
                    self.proj_item_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_item_layer.append(ReLU())
                    self.proj_item_layer.append(torch.nn.Dropout(self.dropout_rate))

                    self.proj_user_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_user_layer.append(ReLU())
                    self.proj_user_layer.append(torch.nn.Dropout(self.dropout_rate))

            self.proj_item_layer.append(Linear(self.hidden_channels, self.out_channels))
            self.proj_user_layer.append(Linear(self.hidden_channels, self.out_channels))

        elif self.two_tower is False:
            self.proj_layer = Sequential()

            if self.n_projs_layer is not None and proj_norm == "LayerNorm":
                for _ in range(self.n_projs_layer):
                    self.proj_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_layer.append(SequentialLayerNorm(self.hidden_channels))
                    self.proj_layer.append(ReLU())
                    self.proj_layer.append(torch.nn.Dropout(self.dropout_rate))

            if self.n_projs_layer is not None and proj_norm == "BatchNorm":
                for _ in range(self.n_projs_layer):
                    self.proj_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_layer.append(SequentialBatchNorm(self.hidden_channels))
                    self.proj_layer.append(ReLU())
                    self.proj_layer.append(torch.nn.Dropout(self.dropout_rate))

            if self.n_projs_layer is not None and proj_norm is None:
                for _ in range(self.n_projs_layer):
                    self.proj_layer.append(Linear(self.hidden_channels, self.hidden_channels))
                    self.proj_layer.append(ReLU())
                    self.proj_layer.append(torch.nn.Dropout(self.dropout_rate))

            self.proj_layer.append(Linear(self.hidden_channels, self.out_channels))
        else:
            raise ValueError("Invalid insert")

    def _lookup_features(self, x: torch.Tensor, embeddings: ModuleDict, feature_cols: list) -> torch.Tensor:
        """
        Encode node features using the corresponding modules.

        Categorical features are passed through embedding layers, while
        numerical features are passed through linear projections.

        Args:
            x (torch.Tensor):
                Input feature matrix of shape `(num_nodes, num_features)`.
            embeddings (ModuleDict):
                Mapping from feature name to encoding module.
            feature_cols (list):
                Ordered list of feature column names.

        Returns:
            torch.Tensor:
                Concatenated encoded features.
        """
        embs = []
        for idx, col in enumerate(feature_cols):
            if isinstance(embeddings[col], Embedding):
                embs.append(embeddings[col](x[:, idx].long()))
            else:
                embs.append(embeddings[col](x[:, idx].unsqueeze(-1)))
        return torch.cat(embs, dim=-1)

    def encoder(self, data: HeteroData) -> dict:
        """
        Encode user and item nodes into latent representations.
        """

        # 1) Initial embeddings
        if not self.add_features:
            user_emb = F.normalize(self.user_emb(data[self.user_name].n_id), dim=-1)
            item_emb = F.normalize(self.item_emb(data[self.item_name].n_id), dim=-1)

        else:
            user_parts = []
            item_parts = []

            if len(self.user_feature_cols) >= 1:
                user_feat_emb = self._lookup_features(
                    data[self.user_name].x,
                    self.user_embeddings,
                    self.user_feature_cols,
                )
                user_parts.append(F.normalize(user_feat_emb, dim=-1))

            if len(self.item_feature_cols) >= 1:
                item_feat_emb = self._lookup_features(
                    data[self.item_name].x,
                    self.item_embeddings,
                    self.item_feature_cols,
                )
                item_parts.append(F.normalize(item_feat_emb, dim=-1))

            if self.num_stochastic_user_features > 0:
                user_stoch_emb = self.user_stochastic_emb(data[self.user_name].n_id)
                user_parts.append(F.normalize(user_stoch_emb, dim=-1))

            if self.num_stochastic_item_features > 0:
                item_stoch_emb = self.item_stochastic_emb(data[self.item_name].n_id)
                item_parts.append(F.normalize(item_stoch_emb, dim=-1))

            user_emb = user_parts[0] if len(user_parts) == 1 else torch.cat(user_parts, dim=-1)
            item_emb = item_parts[0] if len(item_parts) == 1 else torch.cat(item_parts, dim=-1)

        x_dict = {
            self.user_name: user_emb,
            self.item_name: item_emb,
        }

        # 2) Graph convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {
                key: F.dropout(x, p=self.dropout_rate, training=self.training)
                for key, x in x_dict.items()
            }

        # 3) Projection layers
        if self.two_tower:
            x_dict = {
                self.user_name: self.proj_user_layer(x_dict[self.user_name]),
                self.item_name: self.proj_item_layer(x_dict[self.item_name]),
            }
        else:
            x_dict = {
                key: self.proj_layer(x)
                for key, x in x_dict.items()
            }

        return x_dict


    def decoder(self, z_dict: dict, edge_label_index):
        """
        Decode user-item pairs into scalar compatibility scores.

        The score is computed as the dot product between user and item
        latent vectors.

        Args:
            z_dict (dict):
                Dictionary of node embeddings.
            edge_label_index:
                Tensor containing the user and item indices of candidate edges.

        Returns:
            torch.Tensor:
                Score for each candidate edge.
        """
        user_vec = z_dict[self.user_name][edge_label_index[0]]
        item_vec = z_dict[self.item_name][edge_label_index[1]]
        return (user_vec * item_vec).sum(dim=-1)

    def forward(self, data: HeteroData):
        """
        Run the full forward pass of the model.

        In `binary` mode, returns edge scores for link prediction.
        In `triplet` mode, returns normalized user/item embeddings.

        Args:
            data (HeteroData):
                Input heterogeneous graph.

        Returns:
            torch.Tensor | dict:
                - In `binary` mode: tensor of edge scores
                - In `triplet` mode: dictionary of normalized node embeddings
        """
        z_dict = self.encoder(data)

        if self.mode == "binary":
            edge_label_index = data[self.user_name, self.user_to_product, self.item_name].edge_label_index
            score = self.decoder(z_dict, edge_label_index)
            return score
        elif self.mode == "triplet":
            z_dict = {
                self.user_name: F.normalize(z_dict[self.user_name]),
                self.item_name: F.normalize(z_dict[self.item_name]),
            }
            return z_dict
        else:
            raise ValueError("Binary or Triplet")
