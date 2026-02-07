"""
RandomPlayer implements a player that selects random valid moves.
Useful as a baseline or opponent for RL agents.
"""

import numpy as np

from typing import List, Tuple

from reinforcement_learning.players.Player import Player
import random


class RandomPlayer(Player):
    """
    Player that selects random valid moves from the available actions.
    """

    def __init__(self, token: int):
        """
        Initialize a RandomPlayer.

        Args:
            token: 1 for player 1, -1 for player 2
        """
        super().__init__(token)
        self.is_bot = True

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select a random action from the list of valid actions.

        Args:
            state: Current board state (unused)
            valid_actions: List of valid (row, col) actions

        Returns:
            Selected action as a tuple (row, col), or None if no valid actions
        """
        return random.choice(valid_actions) if valid_actions else (-1, -1)
