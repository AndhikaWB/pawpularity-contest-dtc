import json
from copy import deepcopy

import torch
from torch import nn, Tensor
import torchmetrics as tm


class EarlyStopping:
    """Early stopping callback inspired from Keras, since the Lightning version is
    incompatible with Fabric.

    Args:
        monitor (str, optional): Key name to look up on the logs later. The key value
            will be monitored. Defaults to `val_loss`.
        patience (int, optional): How many checks to do before setting `stop_training`
            to True because the value doesn't seem to improve anymore. Defaults to 5.
        mode (str, optional): Set whether `min` or `max` value is better for the
            monitored value. Defaults to `min`.
    
    Examples:
        ```
        logs = {}
        early_stop = EarlyStopping('val_loss')

        for epoch in range(1, epochs + 1):
            for batch, data in enumerate(data_loader):
                ...
                logs['val_loss'] = val_loss

            early_stop.on_epoch_end(epoch, logs)
            if early_stop.stop_training:
                print('Early stopping training...')
                break
        ```
    """

    def __init__(self, monitor: str = 'val_loss', patience: int = 5, mode: str = 'min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.on_train_begin()

    def on_train_begin(self):
        self.wait = 0
        self.best = None
        self.best_epoch = 0
        self.best_logs = None
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: dict):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.__is_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_logs = deepcopy(logs)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True

    def __is_better(self, current, best):
        if best is None:
            return True

        if self.mode == 'min':
            return current < best
        else:
            return current > best


class LossMetric(tm.Metric):
    """Wrap PyTorch loss class as Lightning metric to easily compute average loss at
    the end of each epoch. Note that once wrapped, the loss class will work solely as a
    metric. You should use a separate, unwrapped loss class to calculate the gradient.

    Args:
        loss_class (Module): The PyTorch loss class to wrap (e.g. `BCELoss`).

    Examples:
        ```
        metric = LossMetric(BCELoss)

        for epoch in range(1, epochs + 1):
            for batch, data in enumerate(data_loader):
                ...
                met = metric(preds, labels)

            avg_met = metric.compute().item()
            print(f'Average loss at epoch {epoch} = {avg_met}')
            metric.reset()
        ```
    """

    def __init__(self, loss_cls: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_cls(reduction = 'sum')

        self.add_state('sum_loss', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
        self.add_state('total', default = torch.tensor(0), dist_reduce_fx = 'sum')

    # TODO: Should this be synced with Fabric precision setting?
    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError('Predictions and target must have the same shape')

        with torch.no_grad():
            self.sum_loss += self.loss_fn(preds, target)
            self.total += target.shape[0]

    def compute(self) -> Tensor:
        return self.sum_loss / self.total


class QSave:
    """Quickly save/load plain text or JSON file from local file system."""

    @staticmethod
    def save(obj: dict | str, path: str) -> None:
        with open(path, 'w') as f:
            if type(obj) is dict:
                json.dump(obj, f, indent = 2)
            else:
                f.write(obj)

    @staticmethod
    def load(path: str) -> dict | str:
        with open(path, 'w') as f:
            if path.rsplit('.')[-1] == 'json':
                return json.load(f)
            else:
                return f.read()


class MetricTester:
    """Easily compare a metric value with other value or a threshold."""

    def __init__(self, threshold: float, min_is_better: bool = True):
        self.threshold = threshold
        self.min_is_better = min_is_better
    
    def better_than(self, a: float, b: float):
        """Returns True if value "a" is better than or equal to value "b"."""

        # Why the equal operator? Because we shouldn't change the model unnecessarily
        # If the old model is still equal to the new model, just use the old model
        if self.min_is_better:
            return a <= b
        else:
            return a >= b

    def is_safe(self, value: float):
        """Returns True if the value is still considered normal/safe compared to the
        threshold.
        """

        if self.min_is_better:
            return value < self.threshold
        else:
            return value > self.threshold