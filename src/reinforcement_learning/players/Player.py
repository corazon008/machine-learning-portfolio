"""
Player base class for RL agents and human players in Tic-Tac-Toe.
Defines the interface for action selection and RL hooks.
"""

import numpy as np

from typing import List, Tuple, Any


class Player:
    """
    Abstract base class for a player (agent or human) in Tic-Tac-Toe.
    Subclasses must implement get_action().
    """

    def __init__(self, token: int) -> None:
        """
        Initialize a player with a token (1 or -1).
        Args:
            token: 1 for player 1, -1 for player 2
        """
        if not isinstance(token, int) or token not in [1, -1]:
            raise ValueError("Token must be 1 (player 1) or -1 (player 2).")
        self.token = token
        self.is_bot = True

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select an action from the list of valid actions.
        Must be implemented by subclasses.
        Args:
            state: Current board state (numpy array)
            valid_actions: List of valid (row, col) actions
        Returns:
            Selected action as a tuple (row, col)
        """
        # This method should be overridden by subclasses to implement specific strategies.
        raise NotImplementedError("This method should be overridden by subclasses.")

    # hooks RL (default : no-op)
    def on_action_taken(self, state: Any, action: Any, next_state: Any, reward: float, done: bool) -> None:
        """
        Hook called after an action is taken. Used for RL updates.
        Default is no-op.
        """
        pass

    def on_episode_end(self, final_reward: float) -> None:
        """
        Hook called at the end of an episode. Used for RL updates.
        Default is no-op.
        """
        pass