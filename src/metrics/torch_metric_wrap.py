import torch

from src.metrics.base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        classes = logits.argmax(dim=-1)
        return self.metric(classes, labels)
