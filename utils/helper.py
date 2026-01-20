from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import torch.optim as optim
import yaml


class MyDropout:
    """
    A helper class to manage dropout rates for each layer.
    """
    def __init__(self, dropout_rates: Optional[List[float]] = None):
        # avoid mutable default argument
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.0]
        self.index = 0

    def get(self):
        if self.index > len(self.dropout_rates) - 1:
            return 0
        rate = self.dropout_rates[self.index]
        self.index += 1
        return rate


def find_project_root(markers=("pyproject.toml", ".git")) -> Path:
    for parent in Path.cwd().resolve().parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found")


@dataclass
class Config:
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: optim.Optimizer
    img_size: int
    nb_conv_layers: int
    nb_layers: int
    net_width: int
    dropout_rates: List[float]
    normalization: Dict[str, List[float]]


def load_config(config_path: Path) -> Config:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Map optimizer string to actual optimizer class
    optimizer_map = {'SGD': optim.SGD, 'Adam': optim.Adam, 'RMSprop': optim.RMSprop, }
    if 'optimizer' in config:
        optimizer_name = config['optimizer']
        config['optimizer'] = optimizer_map.get(optimizer_name, optim.Adam)

    return Config(**config)
