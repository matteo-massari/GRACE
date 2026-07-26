import torch
import torch.nn
from torch import autocast
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

from graph_recc_sys.utils.config import DataClass


class Training(DataClass):
    """
    Helper class that manages the training and validation steps for the
    recommendation model.

    This class supports two training regimes:
    - binary training, where the model predicts edge logits and optimizes
      a classification loss
    - contrastive / triplet training, where the model outputs embeddings and
      optimizes a metric-learning loss over anchor-positive-negative triplets

    It stores references to the train/validation subgraphs and loaders, keeps
    track of the current learning rate, and exposes internal methods used by
    the engine during training.

    Attributes:
        train_subgraph (HeteroData):
            Training subgraph used during optimization.
        train_loader (LinkNeighborLoader):
            Mini-batch loader used for training.
        val_loader (LinkNeighborLoader):
            Mini-batch loader used for validation.
        val_subgraph (HeteroData):
            Validation subgraph used for validation-time computations.
        current_lr (float | None):
            Most recently used learning rate.
        loss_fn (torch.nn.Module | None):
            Loss function assigned externally by the training engine.
    """

    def __init__(
        self,
        train_subgraph: HeteroData,
        train_loader: LinkNeighborLoader,
        val_loader: LinkNeighborLoader,
        val_subgraph: HeteroData,
    ) -> None:
        """
        Initialize the training helper.

        Args:
            train_subgraph (HeteroData):
                Training subgraph.
            train_loader (LinkNeighborLoader):
                Loader used for training mini-batches.
            val_loader (LinkNeighborLoader):
                Loader used for validation mini-batches.
            val_subgraph (HeteroData):
                Validation subgraph.
        """
        super().__init__()

        # Training graph
        self.train_subgraph: HeteroData = train_subgraph
        self.train_loader: LinkNeighborLoader = train_loader

        # Validation graph and loader
        self.val_loader: LinkNeighborLoader = val_loader
        self.val_subgraph: HeteroData = val_subgraph

        # Internal state
        self.current_lr: float | None = None
        self.loss_fn: torch.nn.Module | None = None

    def _binary_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        warmup_scheduler,
        scheduler,
        device: torch.device,
    ) -> float:
        """
        Run one full training epoch in binary classification mode.

        In this setting, the model outputs logits for candidate edges and the
        configured binary loss is applied against the batch edge labels.

        Args:
            model (torch.nn.Module):
                Model to train.
            optimizer (torch.optim.Optimizer):
                Optimizer used to update the model parameters.
            scaler (torch.amp.GradScaler):
                Gradient scaler used for mixed precision training.
            warmup_scheduler:
                Warmup scheduler wrapper.
            scheduler:
                Main learning-rate scheduler.
            device (torch.device):
                Device on which training is executed.

        Returns:
            float:
                Average training loss over the epoch.
        """
        model.train()
        train_loss = 0.0
        train_examples = 0
        num_batches = len(self.train_loader)

        for i, sampled_data in enumerate(tqdm(self.train_loader)):
            sampled_data = sampled_data.to(device)
            labels = sampled_data[self.user_name, self.user_to_product, self.item_name].edge_label
            labels = labels.to(dtype=torch.float32, device=device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(sampled_data)
                loss = self.loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = logits.numel()
            train_loss += loss.detach().item() * batch_size
            train_examples += batch_size

            with warmup_scheduler.dampening():
                if i + 1 == num_batches:
                    scheduler.step()

        train_loss = train_loss / max(1, train_examples)
        self.current_lr = optimizer.param_groups[0]["lr"]
        return train_loss

    def _binary_validation(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> float:
        """
        Run one full validation epoch in binary classification mode.

        Args:
            model (torch.nn.Module):
                Model to evaluate.
            device (torch.device):
                Device on which validation is executed.

        Returns:
            float:
                Average validation loss over the epoch.
        """
        val_loss = 0.0
        val_examples = 0

        with torch.no_grad():
            for sampled_data in tqdm(self.val_loader):
                sampled_data = sampled_data.to(device)

                logits = model(sampled_data)
                labels = sampled_data[self.user_name, self.user_to_product, self.item_name].edge_label
                labels = labels.to(dtype=logits.dtype, device=device)

                loss = self.loss_fn(logits, labels)

                batch_size = logits.numel()
                val_loss += loss.item() * batch_size
                val_examples += batch_size

        val_loss = val_loss / max(1, val_examples)
        return val_loss

    def _get_triplets(
        self,
        z_dict: dict[str, torch.Tensor],
        batch: HeteroData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract anchor, positive, and negative embeddings from a batch in
        triplet-training mode.

        For a heterogeneous graph with edge type `(src_type, rel, dst_type)`,
        the method expects the `LinkNeighborLoader` batch to contain:

        - `batch[src_type].src_index`
        - `batch[dst_type].dst_pos_index`
        - `batch[dst_type].dst_neg_index`

        These indices are then used to gather:
        - anchor embeddings from source nodes
        - positive embeddings from destination nodes
        - negative embeddings from destination nodes

        Args:
            z_dict (dict[str, torch.Tensor]):
                Dictionary mapping node types to embedding tensors.
            batch (HeteroData):
                Mini-batch sampled by `LinkNeighborLoader`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing:
                - anchor embeddings
                - positive embeddings
                - negative embeddings
        """
        src_type = self.user_name
        dst_type = self.item_name

        src_index = batch[src_type].src_index
        dst_pos_index = batch[dst_type].dst_pos_index
        dst_neg_index = batch[dst_type].dst_neg_index

        anchor = z_dict[src_type][src_index]
        positive = z_dict[dst_type][dst_pos_index]
        negative = z_dict[dst_type][dst_neg_index]

        return anchor, positive, negative

    def _contrastive_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        warmup_scheduler,
        scheduler,
        device: torch.device,
    ) -> float:
        """
        Run one full training epoch in contrastive / triplet mode.

        In this setting, the model outputs node embeddings. The batch is then
        converted into anchor-positive-negative triplets and the configured
        metric-learning loss is applied.

        Args:
            model (torch.nn.Module):
                Model to train.
            optimizer (torch.optim.Optimizer):
                Optimizer used to update the model parameters.
            scaler (torch.amp.GradScaler):
                Gradient scaler used for mixed precision training.
            warmup_scheduler:
                Warmup scheduler wrapper.
            scheduler:
                Main learning-rate scheduler.
            device (torch.device):
                Device on which training is executed.

        Returns:
            float:
                Average training loss over the epoch.
        """
        model.train()
        train_loss = 0.0
        train_examples = 0
        num_batches = len(self.train_loader)

        for i, sampled_data in enumerate(tqdm(self.train_loader)):
            sampled_data = sampled_data.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                z_dict = model(sampled_data)

                anchor, positive, negative = self._get_triplets(z_dict, sampled_data)

                loss = self.loss_fn(anchor, positive, negative)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = anchor.size(0)
            train_loss += loss.detach().item() * batch_size
            train_examples += batch_size

            with warmup_scheduler.dampening():
                if i + 1 == num_batches:
                    scheduler.step()

        train_loss = train_loss / max(1, train_examples)
        self.current_lr = optimizer.param_groups[0]["lr"]
        return train_loss

    def _contrastive_validation(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> float:
        """
        Run one full validation epoch in contrastive / triplet mode.

        Args:
            model (torch.nn.Module):
                Model to evaluate.
            device (torch.device):
                Device on which validation is executed.

        Returns:
            float:
                Average validation loss over the epoch.
        """
        val_loss = 0.0
        val_examples = 0

        with torch.no_grad():
            for sampled_data in tqdm(self.val_loader):
                sampled_data = sampled_data.to(device)

                z_dict = model(sampled_data)

                anchor, positive, negative = self._get_triplets(z_dict, sampled_data)

                loss = self.loss_fn(anchor, positive, negative)

                batch_size = anchor.size(0)
                val_loss += loss.item() * batch_size
                val_examples += batch_size

        val_loss = val_loss / max(1, val_examples)
        return val_loss
