from pathlib import Path
from typing import List, Optional


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
