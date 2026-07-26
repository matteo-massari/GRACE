import argparse
import logging

import torch
from dotenv import load_dotenv

from graph_recc_sys.engine import NeuralNetworkEngine
from graph_recc_sys.extraction import DatasetManager
from graph_recc_sys.graph_dataset import MarketBasketGraphBuilder
from graph_recc_sys.graph_loader import GraphLoader
from graph_recc_sys.neural_network import RecommenderModel
from graph_recc_sys.utils.hyperparams import HyperParams
from graph_recc_sys.utils.mlflow_config import MlflowConfig
from graph_recc_sys.utils.utils import get_run_name

load_dotenv()


# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

hyperparams = HyperParams.from_yaml()
mlflow_config = MlflowConfig()


# Inizio pipeline
def run_pipeline(training=True, dry_run=False):
    if not dry_run:
        mlflow_config.set_experiment()

    # Scarichiamo i dati
    dataset_manager = DatasetManager(hyperparams)
    dataset_manager.build()
    dataset_manager.download_rec_table()

    # Costruiamo il grafo
    builder = MarketBasketGraphBuilder(dataset_manager=dataset_manager)
    graph_data = builder.graph

    # Creiamo i loader
    loader = GraphLoader(data=graph_data, hyperparams=hyperparams)
    loader.get_loaders()

    # Mandiamo alla GPU
    device = torch.device(f"cuda:{hyperparams.gpu}")
    #device = torch.device(f"cpu")

    # Istanziamo il modello
    model = RecommenderModel(data=graph_data, hyperparams=hyperparams).to(device)

    logging.info(f"Model: {model}\nDevice {device}")

    # Istanziamo l'engine
    engine = NeuralNetworkEngine(loader=loader, hyperparams=hyperparams)

    # Alleniamo
    if training:
        engine.training(
            model=model,
            device=device,
            dry_run=dry_run,
            run_name=get_run_name(model=model, params=hyperparams),
        )
    else:
        model = mlflow_config.get_model(device=device)

        # Inferenza sul test set
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

        previsioni = engine.give_prediction(
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
        help="Esegue la pipeline senza logging su MLflow",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Salta il training e carica il modello specificato negli iperparametri da MLflow",
    )

    args = parser.parse_args()

    run_pipeline(
        training=not args.no_training,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
