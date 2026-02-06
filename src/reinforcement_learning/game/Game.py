import numpy as np


# A simple implementation of a tic-tac-toe game for reinforcement learning purposes.
# Player state is represented as 1 for player 1 and -1 for player 2. Empty cells are represented as 0.
class Game:
    def __init__(self):
        self.state :np.ndarray = np.zeros((3, 3))

    def play(self, player: int, action: tuple) -> bool:
        if self.done:
            raise Exception("Game is already over")

        # Check for out of bounds and invalid player
        if player not in [1, -1] or action[0] not in range(3) or action[1] not in range(3):
            return False

        # Check for legal move
        if self.state[action[0]][action[1]] != 0:
            return False

        # Update the state based on the player's action
        self.state[action[0]][action[1]] = player

        return True

    def check_win(self) -> bool:
        # Check for rows
        for row in self.state:
            if sum(row) == 3 or sum(row) == -3:
                return True

        # Check for columns
        for col in range(3):
            if sum(self.state[:, col]) == 3 or sum(self.state[:, col]) == -3:
                return True

        # Check for diagonals
        if sum(self.state[i][i] for i in range(3)) == 3 or sum(self.state[i][i] for i in range(3)) == -3:
            return True
        if sum(self.state[i][2 - i] for i in range(3)) == 3 or sum(self.state[i][2 - i] for i in range(3)) == -3:
            return True

        return False
