from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dengine.training_strategies.decorators import register_local_training
from dengine.scenarios.decentralized import VanillaDecentralizedSynch
from dengine.training_strategies.local_update_strategy import TrainingEngine


@register_local_training()
class StatefulLRScheduler(TrainingEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduler = None

    def _load_scheduler(self, optimizer: Optimizer) -> Optional[LRScheduler]:
        if not self._scheduler_type:
            return
        if self._scheduler is None:
            assert isinstance(self._scenario, VanillaDecentralizedSynch)
            epochs = self._local_epochs
            self._local_epochs = self._scenario._max_communication_rounds * epochs
            self._scheduler = super()._load_scheduler(optimizer)
            self._local_epochs = epochs
        return self._scheduler
