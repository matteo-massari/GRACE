import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from graph_recc_sys.engine import NeuralNetworkEngine
from graph_recc_sys.extraction import DatasetManager
from graph_recc_sys.utils.hyperparams import HyperParams

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_missing_data_explains_local_setup(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DATA_PATH": tmp}):
            with self.assertRaisesRegex(RuntimeError, "make synthetic-data"):
                DatasetManager(HyperParams.from_yaml()).build()

    def test_export_preserves_synthetic_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = Path(tmp) / "names.csv"
            names.write_text("ids_product,product_name\nP-001,Demo product\n")
            output = Path(tmp) / "new" / "recommendations.csv"
            engine = NeuralNetworkEngine(loader=object(), hyperparams=HyperParams.from_yaml())
            engine.give_prediction(
                torch.tensor([[0]]), {0: "P-001"}, {0: "C-001"}, names, dump=True, dir=output
            )
            row = pd.read_csv(output).iloc[0]
            self.assertEqual(row["ids_customer"], "C-001")
            self.assertEqual(row["ids_product"], "P-001")
            self.assertEqual(row["product_name"], "Demo product")

    def test_generation_is_repeatable_and_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            command = [
                sys.executable,
                str(ROOT / "pipeline/generate_synthetic_database.py"),
                "--output-dir",
                tmp,
            ]
            subprocess.run(command, check=True, capture_output=True)
            prediction = output / "gnn/rec_tables/10/dataset_res_10_features_True.csv"
            self.assertFalse(prediction.exists())
            before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*.csv")}
            prediction.parent.mkdir(parents=True)
            prediction.write_text("actual model output\n")
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(prediction.read_text(), "actual model output\n")
            for path, content in before.items():
                self.assertEqual((output / path).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
