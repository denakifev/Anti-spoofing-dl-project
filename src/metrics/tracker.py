import os
import tempfile

import numpy as np
import pandas as pd
import torch


class MetricTracker:
    """
    Class to aggregate metrics from many batches.
    """

    def __init__(self, *keys, writer=None, metric_funcs=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.metric_funcs = {m.name: m for m in (metric_funcs or [])}

        self._pred_path = os.path.join("/kaggle/working", "preds.raw")
        self._label_path = os.path.join("/kaggle/working", "labels.raw")

        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        for path in [self._pred_path, self._label_path]:
            if os.path.exists(path):
                os.remove(path)

    def update(self, key, value, n=1):
        """
        Update metrics DataFrame with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric.
        """
        return self._data.average[key]

    def accumulate(self, preds, labels=None):
        preds = preds.detach().cpu().to(torch.float32).flatten()
        with open(self._pred_path, "ab") as f:
            f.write(preds.numpy().tobytes())

        if labels is not None:
            labels = labels.detach().cpu().to(torch.float32).flatten()
            with open(self._label_path, "ab") as f:
                f.write(labels.numpy().tobytes())

    def compute_accumulated(self, key, metric_func):
        all_preds, all_labels = self.get_accumulated()
        if len(all_preds) == 0 or len(all_labels) == 0:
            value = float("nan")
        else:
            value = metric_func(all_preds, all_labels)
        self._data.loc[key, "total"] = value
        self._data.loc[key, "counts"] = 1
        self._data.loc[key, "average"] = value
        return value

    def get_accumulated(self):
        preds = torch.frombuffer(
            open(self._pred_path, "rb").read(), dtype=torch.float32
        )
        labels = torch.frombuffer(
            open(self._label_path, "rb").read(), dtype=torch.float32
        )

        return preds, labels

    def result(self):
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        results = {}

        for key in self._data.index:
            metric = self.metric_funcs.get(key, None)
            if metric is not None and getattr(metric, "is_accumulate", False):
                results[key] = self.compute_accumulated(key, metric)
            else:
                results[key] = self._data.average[key]

        return results

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()
