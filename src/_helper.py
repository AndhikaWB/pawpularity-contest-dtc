import json
from copy import deepcopy
import torchmetrics as tm

import torch
from torch import nn, Tensor


# Taken and modified from Keras (keras.callbacks.EarlyStopping)
# The Lightning version is too complicated and incompatible with Fabric
class EarlyStopping:
    def __init__(self, monitor = 'val_loss', patience = 0, mode = 'min'):
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

    def on_epoch_end(self, epoch, logs: dict):
        current = logs.get(self.monitor)
        if current is None: return

        if self._is_better(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_logs = deepcopy(logs)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True

    def _is_better(self, current, best):
        if best is None: return True

        if self.mode == 'min':
            return current < best
        else:
            return current > best


# Wrap PyTorch loss function as Lightning metric
# To automatically get average loss at the end of epoch
class LossWrapper(tm.Metric):
    def __init__(self, loss_cls: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_cls(reduction = 'sum')

        self.add_state('sum_loss', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
        self.add_state('total', default = torch.tensor(0), dist_reduce_fx = 'sum')

    # TODO: Sync with Fabric precision setting?
    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("Predictions and target must have the same shape")

        with torch.no_grad():
            self.sum_loss += self.loss_fn(preds, target)
            self.total += target.shape[0]

    def compute(self) -> Tensor:
        return self.sum_loss / self.total


# Quickly save/load text file
class QSave:
    @staticmethod
    def save(obj: dict | str, path: str) -> None:
        with open(path, 'w') as f:
            if type(obj) == dict:
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