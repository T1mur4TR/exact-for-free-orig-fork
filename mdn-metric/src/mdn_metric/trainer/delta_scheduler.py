import math
from collections import OrderedDict

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder

from ..config import prepare_config


class ExponentDeltaSchedulerCallback(Callback):
    """Exponentially increase DOUCE delta during training."""

    @staticmethod
    def get_default_config(final_delta=1.):
        """Get scheduler parameters."""
        return OrderedDict([
            ("final_delta", final_delta)
        ])

    def __init__(self, num_epochs, *, config=None):
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self._config = prepare_config(self, config)
        self._num_epochs = num_epochs
        self._criterion = None
        self._initial_logdelta = None
        self._final_logdelta = math.log(self._config["final_delta"])

    def on_stage_start(self, runner):
        self._criterion = runner.criterion
        if self._criterion._config["douce_weight"] == 0:
            raise ValueError("Criterion doesn't use DOUCE.")
        self._initial_logdelta = math.log(self._criterion._config["douce_delta"])
        self._epoch = 0

    def on_stage_end(self, runner):
        self._criterion = None
        self._initial_logdelta = None

    def on_epoch_start(self, runner):
        new_delta = math.exp(self._initial_logdelta + (self._final_logdelta - self._initial_logdelta) * self._epoch / self._num_epochs)
        self._criterion._douce._delta = new_delta
        runner.epoch_metrics["_epoch_"]["douce_delta"] = new_delta
        self._epoch += 1
