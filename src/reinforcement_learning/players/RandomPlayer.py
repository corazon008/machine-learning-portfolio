import numpy as np

from typing import List, Tuple

from reinforcement_learning.players.Player import Player
import random


class RandomPlayer(Player):
    def __init__(self, token: int):
        super().__init__(token)
        self.is_bot = True

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int]]) -> Tuple[int]:
        return random.choice(valid_actions) if valid_actions else None
