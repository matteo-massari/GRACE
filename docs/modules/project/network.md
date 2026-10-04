# Recommender Graph Neural Network

## Overview

The module of the **Recommender Graph Neural Network** implements the heart of the system of recommendation based on heterogeneous graph user-item.

The neural network learns latent representations of:

- users;
- products;
- purchase relationships between users and products.

From such embedding, the model can be used both for **binary link prediction** both for **contrastive / metric learning**.

## Objective

The goal of the network is to model purchasing behaviour as a recommendation problem on graph, exploiting:

- user-item graph structure;
- feature of the nodes, if available;
- propagation of information through Graph Neural Network convolutions;
- Final projection in a latent space suitable for ranking or similarity.

## Model architecture

The main model is the class ``RecommenderModel``.

The internal pipeline of the model consists of four blocks:

1. construction of the initial representations of the nodes;
1. propagation through heterogeneous convolutional layer;
1. projection in the final latent space;
1. decode user–item scores or return embeddings.

## Initial representations of the nodes

The model supports two initialization strategies.

### 1. Learnable embeddings

When ``add_features=False``, users and products are represented by embedding matrices learned directly:

{math}`E_u \in \mathbb{R}^{|U| \times d_{in}}, \qquad E_i \in \mathbb{R}^{|I| \times d_{in}}`

Where:

- {math}`|U|` is the number of users;
- {math}`|I|` is the number of products;
- {math}`d_{in}` is the dimensionality of input.

### 2. Embedding based on feature

When ``add_features=True``, Category features are codified via embedding layer and numerical features via linear layers.

The initial representation of a node can be expressed as concatenation of components:

{math}`h_v^{(0)} = \left[ e_{c_1} \Vert e_{c_2} \Vert \dots \Vert W_1 x_{n_1} \Vert W_2 x_{n_2} \Vert \dots \right]`

The model can also concatenate additional stochastic embedding for users and products, useful to increase the representative capacity even in the presence of few informative features.

## Heterogeneous convolution

The network uses a layer stack ``HeteroConv`` with operators ``SAGEConv`` in both directions:

- ``user -> item``
- ``item -> user``

For a target node {math}`v`, GraphSAGE aggregates information from neighbors {math}`\mathcal{N}(v)`:

{math}`m_v^{(l)} = \operatorname{AGGR}\left( \left\{ h_u^{(l)} : u \in \mathcal{N}(v) \right\} \right)`

In implementation, `SAGEConv` uses `project=True` and `normalize=True`.
A schematic form of calculation for each relationship is:

```{math}
\hat h_u = \operatorname{ReLU}(W_p h_u + b_p), \qquad
m_v = \operatorname{AGGR}_{u \in \mathcal{N}(v)} \hat h_u
```

```{math}
h'_v = \operatorname{normalize}(W_r h_v + W_n m_v + b)
```

`HeteroConv` combines the contributions of the reports. Code applies dropout
after each convolution; does not add a further ReLU after `HeteroConv`.
The aggregator is controlled by `aggr`, with `mean` in the configuration provided.

## Projection layers

After convolution, the representations are transformed through a fully connected layer stack.

A typical layer follows the shape:

{math}`z^{(l+1)} = \operatorname{Dropout} \left( \operatorname{ReLU} \left( \operatorname{Norm} \left( W^{(l)} z^{(l)} + b^{(l)} \right) \right) \right)`

The last layer projects in the final latent space:

{math}`z = W_{out} z^{(L)} + b_{out}`

The model can operate in two configurations:

- **single tower**: users and products share the same projection head;
- **two tower**: users and products have two separate heads.

In two-tower configuration:

{math}`z_u = f_u(h_u), \qquad z_i = f_i(h_i)`

## Decoder

### Binary mode

In mode ``binary``, the score between a user {math}`u` and a product {math}`i` is the dot product between end embedding:

{math}`s(u,i) = z_u^T z_i`

This score is then used in a binary loss to distinguish positive and negative edges.

### Triplet mode

In mode ``triplet``, the model directly returns normalized embeddings:

{math}`\tilde{z}_u = \frac{z_u}{\lVert z_u \rVert_2}, \qquad \tilde{z}_i = \frac{z_i}{\lVert z_i \rVert_2}`

These embedding can be used with loss metrics or contrasts.

## Supported losses

The loss is selected by `loss_function`. The entry point set `binary` for
`BCEWithLogitsLoss` and `FocalLoss`and `triplet` for other loss.
The following formulas summarize the objective; reductions and regularization
follow the classes used in the code. In the case of BPR the engine uses `reg_weight=0.1`;
`MultiLoss` use its default values, including `alpha=0.5` and margin 4.0.

### Triplet Margin Loss

{math}`\mathcal{L}_{triplet}(a,p,n) = \max \left( d(a,p) - d(a,n) + m, 0 \right)`

Where? {math}`a` is the anchor, {math}`p` the positive, {math}`n` the negative and {math}`m` the margin.

### BPR Loss

{math}`\mathcal{L}_{BPR} = -\log \sigma(s^+ - s^-)`

with:

{math}`s^+ = a^T p, \qquad s^- = a^T n`

### InfoNCE

{math}`\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\operatorname{sim}(a,p)/\tau)}{\exp(\operatorname{sim}(a,p)/\tau) + \sum_{n \in \mathcal{N}} \exp(\operatorname{sim}(a,n)/\tau)}`

### BCE with logits

{math}`\mathcal{L}_{BCE}(x,y) = - \left[ y \log \sigma(x) + (1-y) \log (1-\sigma(x)) \right]`

### Focal Loss

{math}`\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)`

### MultiLoss

{math}`\mathcal{L}_{tot} = \mathcal{L}_{BPR} + \alpha \cdot \mathcal{L}_{triplet}`

## Data and temporal split

`DatasetManager` read local CSVs. `MarketBasketGraphBuilder` order purchases
by date, preserves the first event of each customer couple–produced and builds
The two relations `buys` and `rev_buys`. The PyG graph is distinguished by Neo4j.

The nodes are `ids_customer` and `ids_product`. In the current configuration
feature customer are `sector`, `potential`, `county`, `creditlimit`, treated
as categories; products use a learned embedding. Customer IDs are not
a category feature of the route `add_features=True`. Mapping and graph
process is saved locally; the cache is reconstructed if changing the
CSV source. Preprocessing changes require explicit reconstruction
cache.

`GraphLoader` order strings for timestamps and use 60% initial for training,
20% later for validation and 20% final for testing, except rounding.
The cut is by number of edges, not by calendar dates: events with the same
timestamps can fall into different splits. Validation and testing both use
the training context, without adding validation purchases to the test.

| Sampler | Behavior |
| --- | --- |
| `neighbor` | `LinkNeighborLoader` time with cutoff prior to the target; requires `pyg-lib` compatible with PyTorch |
| `full` | All nodes, target in chronological order and only edges prior to the first target of the batch; thought for small datasets |

The complete sampler is conservative on the context and does not reproduce sampling
neighbors. Negatives exclude the positives of context and batch; this does not
means that they are products that the customer will never acquire in the future.

## Inference and ranking

Inference uses `model.encoder()` and calculate scores for all couples
customer–product. `similarity` select one of three functions:

```{math}
s(u,i) =
\begin{cases}
-\lVert z_u-z_i\rVert_2 & \text{Euclidean} \\
\dfrac{z_u^Tz_i}{\lVert z_u\rVert_2\lVert z_i\rVert_2} & \text{cosine} \\
z_u^Tz_i & \text{dot}
\end{cases}
```

The path `Euclidean` really uses negative Euclidean distance, not the
dot product. The ranking works on embedding encoder; the
normalization of `forward` triplet is a distinct passage used in training.

Excluded edges receive negative-infinity scores before the top-k. With the graph
preprocessed, the products already bought in the training context are not
candidates. If a customer has less than `k` products available, inference signals
a mistake instead of returning products excluded. The calculation of the complete matrix
of the scores should be considered in the sizing of big datasets.

## Evaluation, tracking and export

| Metrics | Meaning |
| --- | --- |
| Precision@K | Share of the recommendations in the targets |
| Recall@K | Share of targets recovered in recommendations |
| MAP@K | Quality of ordering relevant results |
| Hit ratio@K | Presence of at least one relevant result |
| NDCG@K | Detection with discount for location and normalization |
| Average popularity | Average popularity of recommended products in the training graph |
| Personalization | Diversity between user recommendations lists |

The implementation of metrics is that of classes `LinkPred*` from PyTorch
Geometric. The training loop uses Adam, a cosine scheduler and linear warmup. Every epoch
calculates loss and validation metrics; eventually calculates those of the test.

With active tracking, MLflow receives hyperparameters, checkpoints for epoch and
`final_model`. `--dry-run` only disable tracking. Automatic selection
of the device prefers CUDA, then MPS, then CPU; `--device` allows
the explicit choice.

CSV export takes place in the route `--no-training`, after loading
model from MLflow. Contains `ids_customer`, `ids_product` and `product_name`,
route `DATA_PATH/rec_tables/<k>/dataset_res_<k>_features_<add_features>.csv`.
It does not contain scores or explicit ranks. The context is that of the test: it is not a
retraining on all the historian. To connect the output to Knowledge Graph
you perform `make pipeline-recommendations`.

## Integration into the pipeline

The neural network does not operate in isolation, but is inserted in the complete pipeline of the project:

1. the graph is built from market-basket data and customer/product records;
1. the loader creates temporal splits and mini-batch;
1. the model learns embedding user-item;
1. inference generates top-k recommendations;
1. the evaluation module calculates metrics ranking.

## Main parameters

Among the most relevant hyperparameters of the model:

- ``input_channels``: initial embedding dimension;
- ``hidden_channels``: dimension of convolutional layers;
- ``out_channels``: final dimension of latent space;
- ``n_conv_layer``: number of heterogeneous GraphSAGE layers;
- ``n_projs_layer``: number of projection layers;
- ``dropout_rate``: dropout applied after convolutions and projections;
- ``two_tower``: use of separate heads for users and products;
- ``proj_norm``: normalization in projection layers;
- ``aggr``: aggregation function in SAGEConv layers;
- ``add_features``: use or not of node features.

## When using this network

This architecture is suitable when you want:

- treat recommendation as a problem **link prediction**;
- simultaneously exploit graph and table features;
- use severe temporal splits;
- compare binary training and contrasting training;
- produce reusable latent embedding also in downstream task.

```{note}
The final recommendation quality depends on the network architecture, correct temporal splitting, feature quality, and the negative sampling strategy used during training and evaluation.
```
