import torch
import torch.nn.functional as F
from torch import nn


class AMSoftmaxLoss(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, s: float = 20.0, m: float = 0.9
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor, **batch):
        normalized_features = F.normalize(features, dim=1)
        normalized_weights = F.normalize(self.weight, dim=1)

        cosine = F.linear(normalized_features, normalized_weights)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        cosine_m = cosine - one_hot * self.m

        scaled_logits = self.s * cosine_m

        loss = F.cross_entropy(scaled_logits, labels)

        return {"loss": loss, "logits": scaled_logits}
