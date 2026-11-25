"""Dataset paths: `dataset_path(name)` returns the directory `configs/data_paths.yaml` gives for `name`.

The file is read once, at import. An unknown name raises.
"""

import os

import yaml

# configs/data_paths.yaml of the training directory
_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "configs", "data_paths.yaml")

with open(_CONFIG) as _f:
    DATASETS: dict[str, str] = yaml.safe_load(_f)


def dataset_path(name: str) -> str:
    """`dataset_path("imagenet-1k")` -> the path `configs/data_paths.yaml` gives for that name."""
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; {_CONFIG} defines: {', '.join(sorted(DATASETS))}"
        ) from None
