import numpy as np

from typing import List, Tuple

from reinforcement_learning.players.Player import Player


class RealPlayer(Player):
    def __init__(self, token: int):
        super().__init__(token)
        self.is_bot = False

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int]]) -> Tuple[int]:
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
