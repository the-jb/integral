"""Groups a stream of bucketed samples into same-shape batches.

Batching happens inside the dataset, so the DataLoader runs with `batch_size=None` and never collates.
"""

from collections import defaultdict

import torch
from torch.utils.data import IterableDataset

from .buckets import BucketPlan


class BucketBatcher(IterableDataset):
    """Buffers samples per bucket and emits a stacked batch as soon as one is full.

    Worker and rank splitting are already handled inside the wrapped webdataset pipeline
    (`WebDataset` appends both a nodesplitter and a workersplitter), so each worker just keeps
    its own buffers. Worst-case buffer occupancy is `sum(batch_size[b] - 1)` images.
    """

    def __init__(self, source, plan: BucketPlan):
        super().__init__()
        self.source = source
        self.plan = plan

    def __iter__(self):
        buf = defaultdict(list)
        for sample in self.source:
            b = sample["bucket"]
            buf[b].append(sample["pixel_values"])
            if len(buf[b]) >= self.plan.batch_size[b]:
                yield {"pixel_values": torch.stack(buf.pop(b))}
