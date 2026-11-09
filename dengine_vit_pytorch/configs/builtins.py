import sys
from pathlib import Path
import dengine_vit_pytorch


def get_config(name: str, default_path: str = ""):
    target = Path(sys.prefix) / f'share/dengine-vit-pytorch/configs/{name}'
    if target.exists():
        return target

    root_path = Path(dengine_vit_pytorch.__file__).parent.parent
    return root_path / default_path


class BUILTINS:
    pass
