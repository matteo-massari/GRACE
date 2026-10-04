import argparse
import logging

import torch
from dotenv import load_dotenv

from graph_recc_sys.engine import NeuralNetworkEngine
from graph_recc_sys.extraction import DatasetManager
from graph_recc_sys.graph_dataset import MarketBasketGraphBuilder
from graph_recc_sys.graph_loader import GraphLoader
from graph_recc_sys.neural_network import RecommenderModel
from graph_recc_sys.utils.device import resolve_device
from graph_recc_sys.utils.hyperparams import HyperParams
from graph_recc_sys.utils.mlflow_config import MlflowConfig
from graph_recc_sys.utils.utils import get_run_name

load_dotenv()


# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)


# Pipeline entry point
def run_pipeline(
    training=True, dry_run=False, config=None, epochs=None, device_name="auto", sampler="neighbor", seed=42
):
    torch.manual_seed(seed)
    hyperparams = HyperParams.from_yaml(config)
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        hyperparams.n_epochs = epochs
    hyperparams.mode = (
        "binary" if hyperparams.loss_function in {"BCEWithLogitsLoss", "FocalLoss"} else "triplet"
    )
    device = resolve_device(device_name, hyperparams.gpu)
    logging.info("Selected compute device: %s", device)
    mlflow_config = MlflowConfig()
    if not dry_run:
        mlflow_config.set_experiment()

    # Load the data
    dataset_manager = DatasetManager(hyperparams)
    dataset_manager.build()

    # Build the graph
    builder = MarketBasketGraphBuilder(dataset_manager=dataset_manager)
    graph_data = builder.graph

    # Create the loaders
    loader = GraphLoader(data=graph_data, hyperparams=hyperparams)
    loader.get_loaders(sampler=sampler)

    # Initialize the model
    model = RecommenderModel(data=graph_data, hyperparams=hyperparams).to(device)

    logging.info(f"Model: {model}\nDevice {device}")

    # Initialize the engine
    engine = NeuralNetworkEngine(loader=loader, hyperparams=hyperparams)

    # Train the model
    if training:
        engine.training(
            model=model,
            device=device,
            dry_run=dry_run,
            run_name=get_run_name(model=model, params=hyperparams),
        )
    else:
        model = mlflow_config.get_model(device=device, config_path=config)

        # Run inference on the test set
        results = engine.run_inference(
            model=model,
            subgraph=loader.test_subgraph,
            edge_label_index=loader.test_edge_label_index,
            exclude_links=loader.test_exclude_links,
            device=device,
            k=hyperparams.k,
        )

        results = engine._predict(
            model=model,
            subgraph=loader.test_subgraph,
            exclude_links=loader.test_exclude_links,
            device=device,
            k=hyperparams.k,
        )

        predictions = engine.give_prediction(
            pred_index_mat=results,
            item_inv_map=builder.item_inv_map,
            user_inv_map=builder.user_inv_map,
            product_names_path=dataset_manager.product_names_path,
            edge_label_index=loader.test_edge_label_index,
            pretty_print=False,
            dir=dataset_manager.rec_gnn_path,
            dump=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Run training/testing pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without MLflow logging",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip training and load the configured model from MLflow",
    )

    parser.add_argument("--config", help="Path to the YAML configuration")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument(
        "--sampler",
        choices=["neighbor", "full"],
        default="neighbor",
        help="full: complete temporal contexts for small datasets without pyg-lib",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        training=not args.no_training,
        dry_run=args.dry_run,
        config=args.config,
        epochs=args.epochs,
        device_name=args.device,
        sampler=args.sampler,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
