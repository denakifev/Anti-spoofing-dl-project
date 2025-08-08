import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


class ErrMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        bonafide_probs = logits.softmax(dim=-1)[:, 1]
        return compute_eer(bonafide_probs, labels)


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )

    return frr, far


def compute_eer(pred, ground_truth):
    pred = pred.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    bonafide_scores = pred[ground_truth == 1]
    other_scores = pred[ground_truth == 0]
    frr, far = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer
