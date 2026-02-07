"""
RealPlayer allows a human to play by entering moves via the console.
Prompts for input and validates actions.
"""

import numpy as np

from typing import List, Tuple

from reinforcement_learning.players.Player import Player


class RealPlayer(Player):
    """
    Human player for Tic-Tac-Toe. Prompts for moves via console input.
    """

    def __init__(self, token: int):
        """
        Initialize a RealPlayer.
        Args:
            token: 1 for player 1, -1 for player 2
        """
        super().__init__(token)
        self.is_bot = False

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Prompt the user to enter a move, validate, and return it.
        Args:
            state: Current board state (numpy array)
            valid_actions: List of valid (row, col) actions
        Returns:
            Selected action as a tuple (row, col)
        """
        print(f"Current board state for Player {self.token}:")
        print(state)
        while True:
            try:
                action = input(f"Player {self.token}, enter your move (row and column): ")
                row, col = map(int, action.split())
                if (row, col) in valid_actions:
                    return (row, col)
                else:
                    print("Invalid move. Please enter a valid row and column from the available actions.")
            except ValueError:
                print("Invalid input. Please enter row and column as two integers separated by a space.")
