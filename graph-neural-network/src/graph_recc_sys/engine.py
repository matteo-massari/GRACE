import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_warmup as warmup
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE, info_nce
from torch import amp, device
from torch.optim import Adam, lr_scheduler
from torch_geometric import EdgeIndex
from torch_geometric.data import HeteroData

from graph_recc_sys.engine_modules.evaluation import Evaluation
from graph_recc_sys.engine_modules.inference import Inference
from graph_recc_sys.engine_modules.training import Training
from graph_recc_sys.graph_loader import GraphLoader
from graph_recc_sys.utils.hyperparams import HyperParams
from graph_recc_sys.utils.mlflow_config import MlflowConfig
from graph_recc_sys.utils.recommender_loss import BPRLoss, FocalLoss, MultiLoss


class NeuralNetworkEngine:
    """
    Coordinate training, validation, test-time evaluation, inference, and
    prediction post-processing for a graph-based neural recommendation engine.

    This class acts as the high-level orchestration component of the
    recommendation pipeline. It receives subgraphs, loaders, and label indices
    from a `GraphLoader` instance and delegates the actual operations to
    specialized training, inference, and evaluation components.

    The engine supports multiple training modes, handles optimization utilities
    such as optimizer, learning-rate scheduler, and warmup, logs experiments
    and checkpoints through MLflow, and provides utility methods to decode and
    export generated recommendations.

    Attributes:
        train_subgraph (HeteroData | None):
            Training subgraph used during model optimization.
        train_loader (Any | None):
            Mini-batch loader used during training.
        train_edge_label_index (EdgeIndex | torch.Tensor | None):
            Training edge labels.

        val_loader (Any | None):
            Mini-batch loader used during validation.
        val_subgraph (HeteroData | None):
            Validation subgraph used for inference and evaluation.
        val_edge_label_index (EdgeIndex | torch.Tensor | None):
            Validation ground-truth edges used for metric computation.
        val_exclude_links (EdgeIndex | None):
            Edges to exclude when generating validation recommendations.

        test_subgraph (HeteroData | None):
            Test subgraph used for final inference and evaluation.
        test_edge_label_index (EdgeIndex | torch.Tensor | None):
            Test ground-truth edges used for final metric computation.
        test_exclude_links (EdgeIndex | None):
            Edges to exclude when generating test recommendations.

        params (HyperParams):
            Hyperparameter object controlling model training and inference.

        trainer (Training | None):
            Training helper instance, initialized during training.
        inference (Inference | None):
            Inference helper instance used to generate recommendations.
        evaluator (Evaluation | None):
            Evaluation helper instance used to compute ranking metrics.
    """

    def __init__(self, loader: GraphLoader, hyperparams: HyperParams) -> None:
        """
        Initialize the neural recommendation engine.

        Args:
            loader (GraphLoader):
                Loader object containing subgraphs, loaders, and edge indices
                for train/validation/test.
            hyperparams (HyperParams):
                Hyperparameter configuration for training and inference.
        """
        super().__init__()

        # Training graph
        self.train_subgraph: HeteroData | None = getattr(loader, "train_subgraph", None)
        self.train_loader: Any | None = getattr(loader, "train_loader", None)
        self.train_edge_label_index: EdgeIndex | torch.Tensor | None = getattr(
            loader, "train_edge_label_index", None
        )

        # Validation graph
        self.val_loader: Any | None = getattr(loader, "val_loader", None)
        self.val_subgraph: HeteroData | None = getattr(loader, "val_subgraph", None)
        self.val_edge_label_index: EdgeIndex | torch.Tensor | None = getattr(
            loader, "val_edge_label_index", None
        )
        self.val_exclude_links: EdgeIndex | None = getattr(loader, "val_exclude_links", None)

        # Test graph
        self.test_subgraph: HeteroData | None = getattr(loader, "test_subgraph", None)
        self.test_edge_label_index: EdgeIndex | torch.Tensor | None = getattr(
            loader, "test_edge_label_index", None
        )
        self.test_exclude_links: EdgeIndex | None = getattr(loader, "test_exclude_links", None)

        self.params: HyperParams = hyperparams

        # Helper components
        self.trainer: Training | None = None
        self.inference: Inference | None = None
        self.evaluator: Evaluation | None = None

    def training(
        self,
        model: torch.nn.Module,
        dry_run: bool,
        device: device,
        run_name: str,
    ) -> None:
        """
        Train the model, run validation at each epoch, and log metrics and
        checkpoints to MLflow.

        Depending on the configured loss function, the engine automatically
        switches between:
        - `triplet` mode for metric-learning / contrastive losses
        - `binary` mode for edge classification losses

        Supported loss functions:
        - `TripletMarginLoss`
        - `BPRLoss`
        - `InfoNCE`
        - `MultiLoss`
        - `BCEWithLogitsLoss`
        - `FocalLoss`

        Args:
            model (torch.nn.Module):
                PyTorch model to train.
            dry_run (bool):
                If True, no artifacts or metrics are logged to MLflow.
            device (device):
                PyTorch device used for training and validation.
            run_name (str):
                Name of the MLflow run.

        Returns:
            None

        Raises:
            ValueError:
                If `self.params.loss_function` is invalid.
            ValueError:
                If `self.params.mode` becomes invalid during training.
        """
        self.trainer = Training(
            train_subgraph=self.train_subgraph,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            val_subgraph=self.val_subgraph,
        )
        self.inference = Inference()
        self.evaluator = Evaluation()

        if dry_run:
            logging.info("=== Dry run ===")

        optimizer = Adam(model.parameters(), lr=self.params.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.params.n_epochs,
            eta_min=self.params.eta_min,
        )
        warmup_scheduler = warmup.LinearWarmup(
            optimizer=optimizer,
            warmup_period=self.params.warmup_period * len(self.train_loader),
        )
        scaler = amp.GradScaler(device=device.type, enabled=device.type == "cuda")

        loss_name = self.params.loss_function
        logging.info(f"Loss function: {loss_name}")
        logging.info(f"Training starting with: {model.__class__.__name__}")
        logging.info(f"Type of distance: {self.params.similarity}")

        if loss_name == "TripletMarginLoss":
            self.params.mode = "triplet"
            self.trainer.loss_fn = torch.nn.TripletMarginLoss(margin=self.params.margin)

        elif loss_name == "BPRLoss":
            self.params.mode = "triplet"
            self.trainer.loss_fn = BPRLoss(reg_weight=0.1)

        elif loss_name == "InfoNCE":
            self.params.mode = "triplet"
            self.trainer.loss_fn = InfoNCE(temperature=0.2)

        elif loss_name == "MultiLoss":
            self.params.mode = "triplet"
            self.trainer.loss_fn = MultiLoss()

        elif loss_name == "BCEWithLogitsLoss":
            self.params.mode = "binary"
            self.trainer.loss_fn = torch.nn.BCEWithLogitsLoss()

        elif loss_name == "FocalLoss":
            self.params.mode = "binary"
            self.trainer.loss_fn = FocalLoss(
                gamma=4.5,
                alpha=None,
                reduction="mean",
                task_type="binary",
                num_classes=None,
            )

        else:
            raise ValueError(
                f"Invalid params.loss_function={loss_name!r}. "
                "Expected one of: "
                "['TripletMarginLoss', 'BPRLoss', 'InfoNCE', 'MultiLoss', "
                "'BCEWithLogitsLoss', 'FocalLoss']"
            )

        with MlflowConfig.mlflow_run(dry_run=dry_run, run_name=run_name):
            if not dry_run:
                mlflow.log_params(asdict(self.params))

            for epoch in range(1, self.params.n_epochs + 1):
                if self.params.mode == "triplet":
                    model.train()
                    train_loss = self.trainer._contrastive_training(
                        model, optimizer, scaler, warmup_scheduler, scheduler, device
                    )

                    model.eval()
                    val_loss = self.trainer._contrastive_validation(model=model, device=device)

                elif self.params.mode == "binary":
                    model.train()
                    train_loss = self.trainer._binary_training(
                        model, optimizer, scaler, warmup_scheduler, scheduler, device
                    )

                    model.eval()
                    val_loss = self.trainer._binary_validation(model=model, device=device)

                else:
                    raise ValueError(
                        f"Invalid params.mode={self.params.mode!r}. "
                        "Expected one of: ['triplet', 'binary']"
                    )

                if not dry_run:
                    model_info = mlflow.pytorch.log_model(
                        pytorch_model=model,
                        serialization_format="pickle",
                        name=f"checkpoint_{epoch}",
                        step=epoch,
                    )

                    mlflow.log_metric(
                        key="train_loss",
                        value=train_loss,
                        step=epoch,
                        model_id=model_info.model_id,
                    )

                    mlflow.log_metric(
                        key="val_loss",
                        value=val_loss,
                        step=epoch,
                        model_id=model_info.model_id,
                    )

                    mlflow.log_metric(
                        key="current_lr",
                        value=self.trainer.current_lr,
                        step=epoch,
                        model_id=model_info.model_id,
                    )

                pred_index_mat = self.inference.test_epoch_dist(
                    model=model,
                    subgraph=self.val_subgraph,
                    exclude_links=self.val_exclude_links,
                    similarity=self.params.similarity,
                    device=device,
                    k=self.params.k,
                )

                res = self.evaluator.evaluate_epoch(
                    pred_index_mat=pred_index_mat,
                    popularity_subgraph=self.train_subgraph,
                    edge_label_index=self.val_edge_label_index,
                    device=device,
                    k=self.params.k,
                )

                if not dry_run:
                    mlflow.log_metrics(
                        res,
                        step=epoch,
                        model_id=model_info.model_id,
                    )

                logging.info(
                    f"Epoch {epoch:03d} - train Loss: {train_loss:.4f} "
                    f"- val Loss: {val_loss:.4f} - current lr: {self.trainer.current_lr:.8f} - {res}"
                )

            pred_index_mat = self.inference.test_epoch_dist(
                model=model,
                subgraph=self.test_subgraph,
                exclude_links=self.test_exclude_links,
                similarity=self.params.similarity,
                device=device,
                k=self.params.k,
            )

            res = self.evaluator.evaluate_epoch(
                pred_index_mat=pred_index_mat,
                popularity_subgraph=self.train_subgraph,
                edge_label_index=self.test_edge_label_index,
                device=device,
                k=self.params.k,
            )

            test_res = {
                "test/precision_at_k": res["Precision_at_K"],
                "test/recall_at_k": res["Recall_at_K"],
                "test/map_at_k": res["MAP_at_K"],
                "test/hit_at_k": res["Hit_at_k"],
                "test/ndcg_at_k": res["NDCG_at_K"],
                "test/avg_popularity": res["prev_avg_pop_metric"],
                "test/personalization": res["personalization_score"],
            }

            if not dry_run:
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    serialization_format="pickle",
                    name="final_model",
                )

                mlflow.log_metrics(
                    test_res,
                    model_id=model_info.model_id,
                )

            logging.info(f"Final test metrics - {test_res}")

    def _predict(
        self,
        model: torch.nn.Module,
        subgraph: HeteroData,
        exclude_links: EdgeIndex,
        device: device,
        k: int,
    ) -> torch.Tensor:
        """
        Generate top-k recommendations without computing evaluation metrics.

        Args:
            model (torch.nn.Module):
                Trained PyTorch model.
            subgraph (HeteroData):
                Graph on which inference is performed.
            exclude_links (EdgeIndex):
                Edges to exclude from candidate recommendations.
            device (device):
                PyTorch device used for inference.
            k (int):
                Number of recommendations to generate.

        Returns:
            torch.Tensor:
                Matrix of predicted item indices with shape `(n_users, k)`.
        """
        if self.inference is None:
            self.inference = Inference()

        model.eval()
        with torch.inference_mode():
            pred_index_mat = self.inference.test_epoch_dist(
                model=model,
                subgraph=subgraph,
                exclude_links=exclude_links,
                similarity=self.params.similarity,
                device=device,
                k=k,
            )

        return pred_index_mat

    def run_inference(
        self,
        model: torch.nn.Module,
        subgraph: HeteroData,
        edge_label_index: EdgeIndex,
        exclude_links: EdgeIndex,
        device: device,
        k: int,
        popularity_subgraph: HeteroData | None = None,
    ) -> dict[str, float]:
        """
        Run inference on a given subgraph and return prediction metrics.

        Args:
            model (torch.nn.Module):
                Trained PyTorch model.
            subgraph (HeteroData):
                Graph on which inference is performed.
            edge_label_index (EdgeIndex):
                Ground-truth edges used for evaluation.
            exclude_links (EdgeIndex):
                Edges to exclude from prediction candidates
                (for example training edges).
            device (device):
                PyTorch device used for inference.
            k (int):
                Number of top-k recommendations.
            popularity_subgraph (HeteroData | None):
                Graph used to compute popularity-based metrics.
                If None, `self.train_subgraph` is used.

        Returns:
            dict[str, float]:
                Dictionary containing the computed evaluation metrics.
        """
        if self.inference is None:
            self.inference = Inference()
        if self.evaluator is None:
            self.evaluator = Evaluation()

        if popularity_subgraph is None:
            popularity_subgraph = self.train_subgraph

        model.eval()
        with torch.inference_mode():
            pred_index_mat = self.inference.test_epoch_dist(
                model=model,
                subgraph=subgraph,
                exclude_links=exclude_links,
                similarity=self.params.similarity,
                device=device,
                k=k,
            )

            metrics = self.evaluator.evaluate_epoch(
                pred_index_mat=pred_index_mat,
                popularity_subgraph=popularity_subgraph,
                edge_label_index=edge_label_index,
                device=device,
                k=k,
            )

        logging.info(f"Inference metrics: {metrics}")
        return metrics

    def give_prediction(
        self,
        pred_index_mat: torch.Tensor,
        item_inv_map: dict[int, Any],
        user_inv_map: dict[int, Any],
        product_names_path: str | Path,
        user_idx_list: list[int] | torch.Tensor | None = None,
        edge_label_index: EdgeIndex | torch.Tensor | None = None,
        pretty_print: bool = False,
        dump: bool = False,
        dir: str | Path | None = None,
    ) -> dict[str, dict[str, dict[str, str | None]]]:
        """
        Decode model predictions into original customer/product identifiers and,
        if provided, attach the products actually purchased in the ground truth.

        If `dump=True`, the method also exports the recommendations to a CSV file.

        Output structure::

            {
                customer_id: {
                    "bought": {
                        product_id: product_name,
                        ...
                    },
                    "recommended": {
                        product_id: product_name,
                        ...
                    }
                },
                ...
            }

        Args:
            pred_index_mat (torch.Tensor):
                Matrix of predicted item indices with shape `(n_users, k)`.
            item_inv_map (dict[int, Any]):
                Reverse mapping from internal item indices to original product ids.
            user_inv_map (dict[int, Any]):
                Reverse mapping from internal user indices to original customer ids.
            product_names_path (str | Path):
                Path to the CSV file containing product ids and product names.
            user_idx_list (list[int] | torch.Tensor | None):
                Optional list of global user indices corresponding to the rows of
                `pred_index_mat`. If None, row indices are treated as global ids.
            edge_label_index (EdgeIndex | torch.Tensor | None):
                Optional ground-truth user-item edges to attach as `"bought"`.
            pretty_print (bool):
                If True, print the decoded output as formatted JSON.
            dump (bool):
                If True, export recommendations to CSV.
            dir (str | Path | None):
                Output CSV path used when `dump=True`.

        Returns:
            dict[str, dict[str, dict[str, str | None]]]:
                Nested dictionary containing bought and recommended products
                for each customer.

        Raises:
            ValueError:
                If `dump=True` and no output path is provided.
            KeyError:
                If a predicted user or item index is not found in the provided
                inverse mappings.
        """

        def normalize_id(x: Any) -> str:
            if isinstance(x, (bytes, bytearray)):
                x = x.decode("utf-8")
            value = str(x).strip()
            return value.zfill(10) if value.isdecimal() else value

        def normalize_name(x: Any) -> Any:
            if isinstance(x, (bytes, bytearray)):
                x = x.decode("utf-8")
            return x

        def to_int(x: Any) -> int:
            return x.item() if hasattr(x, "item") else int(x)

        if dump and not dir:
            raise ValueError("If dump=True, you must provide dir.")

        product_df = pd.read_csv(product_names_path, dtype={"ids_product": str})
        product_df["ids_product"] = product_df["ids_product"].apply(normalize_id)
        product_df["product_name"] = product_df["product_name"].apply(normalize_name)

        product_name_map = dict(zip(product_df["ids_product"], product_df["product_name"]))

        output: dict[str, dict[str, dict[str, str | None]]] = {}
        rows: list[dict[str, str | None]] = []

        # Part 1: recommendations
        for local_idx, user_row in enumerate(pred_index_mat):
            if user_idx_list is not None:
                user_idx = to_int(user_idx_list[local_idx])
            else:
                user_idx = local_idx

            if user_idx not in user_inv_map:
                raise KeyError(
                    f"Global user index {user_idx} (local row {local_idx}) not found in user_inv_map."
                )

            customer_id = normalize_id(user_inv_map[user_idx])
            recommended: dict[str, str | None] = {}

            for idx in user_row:
                item_idx = to_int(idx)

                if item_idx not in item_inv_map:
                    raise KeyError(f"Product index {item_idx} not found in item_inv_map.")

                product_id = normalize_id(item_inv_map[item_idx])
                product_name = product_name_map.get(product_id)

                recommended[product_id] = product_name

                if dump:
                    rows.append(
                        {
                            "ids_customer": customer_id,
                            "ids_product": product_id,
                            "product_name": product_name,
                        }
                    )

            output[customer_id] = {
                "bought": {},
                "recommended": recommended,
            }

        # Part 2: ground truth purchases
        if edge_label_index is not None:
            user_indices = edge_label_index[0]
            item_indices = edge_label_index[1]

            for user_idx_tensor, item_idx_tensor in zip(user_indices, item_indices):
                user_idx = to_int(user_idx_tensor)
                item_idx = to_int(item_idx_tensor)

                if user_idx not in user_inv_map or item_idx not in item_inv_map:
                    continue

                customer_id = normalize_id(user_inv_map[user_idx])
                product_id = normalize_id(item_inv_map[item_idx])
                product_name = product_name_map.get(product_id)

                if customer_id not in output:
                    output[customer_id] = {"bought": {}, "recommended": {}}

                output[customer_id]["bought"][product_id] = product_name

        if dump:
            result_df = pd.DataFrame(rows, columns=["ids_customer", "ids_product", "product_name"])
            Path(dir).parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(path_or_buf=dir, index=False)

        if pretty_print:
            print(json.dumps(output, indent=2, ensure_ascii=False))

        return output
