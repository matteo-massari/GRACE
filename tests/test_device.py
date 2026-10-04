import unittest
from unittest.mock import patch

from graph_recc_sys.utils.device import resolve_device


class DeviceTests(unittest.TestCase):
    def test_auto_priority_and_cpu_override(self):
        for cuda, mps, expected in [(True, True, "cuda:1"), (False, True, "mps"), (False, False, "cpu")]:
            with (
                self.subTest(cuda=cuda, mps=mps),
                patch("torch.cuda.is_available", return_value=cuda),
                patch("torch.cuda.device_count", return_value=2),
                patch("torch.backends.mps.is_available", return_value=mps),
            ):
                self.assertEqual(str(resolve_device("auto", gpu=1)), expected)
                self.assertEqual(str(resolve_device("cpu")), "cpu")

    def test_unavailable_explicit_devices_fail(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            for name in ["cuda", "cuda:0", "mps"]:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unavailable"):
                    resolve_device(name)

    def test_invalid_backend_and_cuda_index(self):
        for name in ["invalid", "meta", "cpu:1", "cuda:-1"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_device(name)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            self.assertEqual(str(resolve_device("cuda")), "cuda:0")
            with self.assertRaisesRegex(ValueError, "index"):
                resolve_device("cuda:1")


class ConfigurationCompatibilityTests(unittest.TestCase):
    def test_legacy_yaml_aliases_and_conflicts(self):
        import tempfile
        from pathlib import Path

        import yaml

        from graph_recc_sys.utils.hyperparams import HyperParams

        source = Path(__file__).resolve().parents[1] / "graph-neural-network/smoke.yaml"
        data = yaml.safe_load(source.read_text())
        graph = data["hyperparameters"]["graph"]
        for old, new in [
            ("n_clienti", "n_customers"),
            ("n_mesi", "n_months"),
            ("n_connessioni_minime", "min_connections"),
        ]:
            graph[old] = graph.pop(new)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "legacy.yaml"
            config.write_text(yaml.safe_dump(data))
            params = HyperParams.from_yaml(config)
            self.assertEqual(params, HyperParams.from_yaml(source))
            graph["n_months"] = 12
            config.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                HyperParams.from_yaml(config)
