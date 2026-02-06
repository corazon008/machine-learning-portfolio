import numpy as np


# A simple implementation of a tic-tac-toe game for reinforcement learning purposes.
# Player state is represented as 1 for player 1 and -1 for player 2. Empty cells are represented as 0.
class Game:
    def __init__(self):
        self._state :np.ndarray = np.zeros((3, 3))

    def play(self, player: int, action: tuple) -> bool:
        # Check for out of bounds and invalid player
        if player not in [1, -1] or action[0] not in range(3) or action[1] not in range(3):
            return False

        # Check for legal move
        if self._state[action[0]][action[1]] != 0:
            return False

        # Update the _state based on the player's action
        self._state[action[0]][action[1]] = player

        return True

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def get_valid_actions(self) -> list:
        return [(i, j) for i in range(3) for j in range(3) if self._state[i][j] == 0]

    def check_win(self) -> bool:
        # Check for rows
        for row in self._state:
            if sum(row) == 3 or sum(row) == -3:
                return True

        # Check for columns
        for col in range(3):
            if sum(self._state[:, col]) == 3 or sum(self._state[:, col]) == -3:
                return True

        # Check for diagonals
        if sum(self._state[i][i] for i in range(3)) == 3 or sum(self._state[i][i] for i in range(3)) == -3:
            return True
        if sum(self._state[i][2 - i] for i in range(3)) == 3 or sum(self._state[i][2 - i] for i in range(3)) == -3:
            return True

        return False

    def check_draw(self) -> bool:
        return all(cell != 0 for row in self._state for cell in row) and not self.check_win()
