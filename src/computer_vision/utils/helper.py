from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import torch.optim as optim
import yaml


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
