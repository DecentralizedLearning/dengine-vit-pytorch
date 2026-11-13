import importlib
from typing import Optional, Type

import torch
from torch.utils.data import Subset

import vit_pytorch

from dengine.models.decorators import register_model
from dengine.models.utils import get_unique_targets, get_image_channels, get_image_size
from dengine.models import ModuleBase


@register_model()
class VitPytorchModelBase(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        cls_name: str,
        *args,
        image_size: Optional[int] = None,
        in_chans: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        num_channels = in_chans or get_image_channels(dataset)
        image_size = image_size or get_image_size(dataset)
        num_classes = num_classes or len(get_unique_targets(dataset))

        vit_kwargs = {k[15:]: v for k, v in kwargs.items() if k.startswith("__pytorch_vit__")}

        module_name, _, attr = cls_name.rpartition(".")
        model_cls = getattr(importlib.import_module(module_name) if module_name else vit_pytorch, attr)

        self._model = model_cls(
            num_classes=num_classes,
            channels=num_channels,
            img_size=image_size,
            **vit_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def create_model(model: Type, **kwargs):
    return {
        "client.local_model.target": VitPytorchModelBase,
        "client.local_model.arguments.cls_name": f"{model.__module__}.{model.__name__}",
        **{f"client.local_model.arguments.__pytorch_vit__{key}": value for key, value in kwargs.items()}
    }
