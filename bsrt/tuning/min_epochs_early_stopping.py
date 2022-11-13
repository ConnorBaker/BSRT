from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer import Trainer
from typing_extensions import Literal


class MinEpochsEarlyStopping(EarlyStopping):
    """
    A custom early stopping callback that doesn't stop training if the minimum number of epochs
    hasn't been reached yet. This is useful for when we want to train for a longer time to see if
    the model can recover from a bad initialization.

    Patience is incremented by 1 if the minimum number of epochs hasn't been reached yet, ensuring
    that we don't stop training prematurely. It is reset to the original value once the minimum
    number of epochs has been reached. In this way, patience does not start until we have reached
    the minimum number of epochs.

    When the minimum number of epochs has been reached, in addition to patience being reset to its
    original value, the divergence threshold is added to the callback and the wait counter is
    reset to zero.
    """

    def __init__(
        self,
        monitor: str,
        min_delta: float,
        patience: int,
        min_epochs: int,
        mode: Literal["min", "max"],
        divergence_threshold: float,
        verbose: bool = False,
    ):
        """
        Args:
            divergence_threshold (float): The divergence threshold for the metric to be monitored.
            monitor (str): The metric to be monitored.
            patience (int): The number of epochs to wait for the metric to be monitored to improve.
            min_epochs (int): The minimum number of epochs to train for.
            min_delta (float): The minimum change in the monitored metric to qualify as an
                improvement.
            mode (Literal["min", "max"]): Whether the monitored metric should be increasing or
                decreasing.
            verbose (bool, optional): Whether to print messages. Defaults to False.
        """
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            verbose=verbose,
        )
        self.min_epochs = min_epochs
        self._divergence_threshold = divergence_threshold

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch < self.min_epochs:
            # Reset the wait counter to 0
            self.wait_count = 0
        elif trainer.current_epoch == self.min_epochs:
            # Reset the wait counter to 0
            self.wait_count = 0
            # Add the divergence threshold attribute now that we've reached the minimum number of
            # epochs
            self.divergence_threshold = self._divergence_threshold

        self._run_early_stopping_check(trainer)
