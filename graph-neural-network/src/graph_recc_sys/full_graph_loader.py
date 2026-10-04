"""Small-data temporal batches without optional native sampling extensions."""

import math

import torch


class FullGraphLinkLoader:
    """Keep all nodes and only context edges older than each batch's earliest target.

    This conservative CPU/demo path does not approximate neighbor sampling: it
    uses the complete eligible context. Targets are batched chronologically.
    """

    def __init__(
        self, data, num_neighbors, edge_label_index, edge_label_time, batch_size, neg_sampling, **kwargs
    ):
        self.data = data
        self.edge_type, self.targets = edge_label_index
        self.times = edge_label_time
        self.batch_size = batch_size
        self.mode = neg_sampling["mode"]

    def __len__(self):
        return math.ceil(self.targets.size(1) / self.batch_size)

    def __iter__(self):
        user, _, item = self.edge_type
        reverse = next(t for t in self.data.edge_types if t[0] == item and t[2] == user)
        order = self.times.argsort(stable=True)
        num_items = self.data[item].num_nodes
        for start in range(0, order.numel(), self.batch_size):
            ids = order[start : start + self.batch_size]
            targets = self.targets[:, ids]
            mask = self.data[self.edge_type].time <= self.times[ids].min()
            context_ids = mask.nonzero(as_tuple=True)[0]
            batch = self.data.edge_subgraph({self.edge_type: context_ids, reverse: context_ids})
            # Avoid known positives and all positive targets in this batch.
            known = torch.cat([batch[self.edge_type].edge_index, targets], dim=1)
            negatives = []
            for uid in targets[0]:
                candidates = torch.ones(num_items, dtype=torch.bool)
                candidates[known[1, known[0] == uid]] = False
                candidates = candidates.nonzero(as_tuple=True)[0]
                if candidates.numel() == 0:
                    raise ValueError("No negative product available for this customer")
                negatives.append(candidates[torch.randint(candidates.numel(), (1,))].item())
            negatives = torch.tensor(negatives, dtype=torch.long)
            if self.mode == "triplet":
                batch[user].src_index = targets[0]
                batch[item].dst_pos_index = targets[1]
                batch[item].dst_neg_index = negatives
            else:
                batch[self.edge_type].edge_label_index = torch.cat(
                    [targets, torch.stack([targets[0], negatives])], dim=1
                )
                batch[self.edge_type].edge_label = torch.cat(
                    [torch.ones(ids.numel()), torch.zeros(ids.numel())]
                )
            yield batch
