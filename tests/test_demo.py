import random
import unittest
from collections import Counter

import torch
from torch_geometric.data import HeteroData

from graph_recc_sys.engine_modules.inference import Inference
from graph_recc_sys.full_graph_loader import FullGraphLinkLoader
from graph_recc_sys.utils.hyperparams import HyperParams
from pipeline.generate_synthetic_database import (
    SEED,
    customer_rows,
    product_rows,
    purchase_rows,
    recommendation_rows,
)


class DemoTests(unittest.TestCase):
    def test_synthetic_integrity(self):
        rng = random.Random(SEED)
        customers, products = customer_rows(rng), product_rows()
        purchases = purchase_rows(rng, customers, products)
        customer_ids = {r["ids_customer"] for r in customers}
        product_ids = {r["level5id"] for r in products}
        self.assertEqual(len(customer_ids), 100)
        self.assertEqual(len(product_ids), 60)
        counts = Counter(p["ids_customer"] for p in purchases)
        self.assertEqual(set(counts), customer_ids)
        self.assertTrue(all(10 <= n <= 100 for n in counts.values()))
        self.assertTrue(all(p["level5id"] in product_ids for p in purchases))
        seen = {(p["ids_customer"], p["level5id"]) for p in purchases}
        recs = recommendation_rows(customers, products, purchases)
        self.assertEqual(len(recs), 1000)
        self.assertTrue(all((r["ids_customer"], r["ids_product"]) not in seen for r in recs))
        self.assertTrue(all(r["source"] == "placeholder_not_gnn" for r in recs))

    def test_default_config_from_root(self):
        self.assertEqual(HyperParams.from_yaml().n_epochs, 50)

    def test_distance_is_not_dot_product(self):
        query = torch.tensor([[1.0, 0.0]])
        items = torch.tensor([[10.0, 0.0], [1.0, 0.1]])
        self.assertEqual(Inference.mips_search(query, items, 1, "euclidean").item(), 1)
        self.assertEqual(Inference.mips_search(query, items, 1, "dot").item(), 0)

    def test_excluded_items_never_returned(self):
        query, items = torch.ones(1, 2), torch.eye(2)
        excluded = torch.tensor([[0], [0]])
        self.assertEqual(Inference.mips_search(query, items, 1, "cosine", excluded).item(), 1)
        with self.assertRaises(ValueError):
            Inference.mips_search(query, items, 2, "cosine", excluded)

    def test_temporal_batches_and_negatives(self):
        data = HeteroData()
        edge, rev = ("u", "buys", "i"), ("i", "rev_buys", "u")
        data["u"].num_nodes, data["i"].num_nodes = 1, 4
        data["u"].n_id, data["i"].n_id = torch.arange(1), torch.arange(4)
        data[edge].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
        data[rev].edge_index = data[edge].edge_index.flip(0)
        data[edge].time = data[rev].time = torch.tensor([10, 20, 30])
        for mode in ["triplet", "binary"]:
            loader = FullGraphLinkLoader(
                data, [], (edge, data[edge].edge_index), data[edge].time - 1, 1, {"mode": mode}
            )
            for idx, batch in enumerate(loader):
                self.assertTrue((batch[edge].time < data[edge].time[idx]).all())
                self.assertTrue(torch.equal(batch[rev].edge_index, batch[edge].edge_index.flip(0)))
                neg = batch["i"].dst_neg_index if mode == "triplet" else batch[edge].edge_label_index[1, 1:]
                self.assertTrue((neg > idx).all())


if __name__ == "__main__":
    unittest.main()
