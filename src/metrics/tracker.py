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

        self._preds = []
        self._labels = []

        self.reset()

    def reset(self):
        """
        Reset all metrics after epoch end.
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0
        self._preds = []
        self._labels = []

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
        self._preds.append(preds)
        if labels is not None:
            self._labels.append(labels)

    def compute_accumulated(self, key, metric_func):
        all_preds, all_labels = (
            torch.cat(self._preds, dim=0).cpu(),
            torch.cat(self._labels, dim=0).cpu(),
        )

        if len(all_preds) == 0 or len(all_labels) == 0:
            value = float("nan")
        else:
            value = metric_func(all_preds, all_labels)
        self._data.loc[key, "total"] = value
        self._data.loc[key, "counts"] = 1
        self._data.loc[key, "average"] = value
        return value

    def keys(self):
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (Index): all metric names in the table.
        """
        return self._data.total.keys()

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
