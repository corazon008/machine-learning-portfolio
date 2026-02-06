from reinforcement_learning.players.Player import Player
import random

class RandomPlayer(Player):
    def __init__(self, token: int):
        super().__init__(token)
        self.is_bot = True

    def get_action(self, state)-> tuple:
        empty_cells = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]
        return random.choice(empty_cells) if empty_cells else None