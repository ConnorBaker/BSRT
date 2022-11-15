from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from syne_tune.report import Reporter


@dataclass
class SyneTuneReporterCallback(Callback):
    metric_names: Union[Sequence[str], Mapping[str, str]]
    reporter: Reporter

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = {}
        if isinstance(self.metric_names, Sequence):
            for metric_name in self.metric_names:
                metric = trainer.callback_metrics[metric_name]
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()

                metrics[metric_name] = metric
        else:
            for metric_name, new_name in self.metric_names.items():
                metric = trainer.callback_metrics[metric_name]
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()

                metrics[new_name] = metric

        self.reporter(epoch=trainer.current_epoch, **metrics)
