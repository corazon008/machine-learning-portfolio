"""
Game module for Tic-Tac-Toe environment used in reinforcement learning experiments.
Provides the Game class for board state management, move validation, and win/draw checking.
"""
import numpy as np
from typing import List, Tuple

# A simple implementation of a tic-tac-toe game for reinforcement learning purposes.
# Player state is represented as 1 for player 1 and -1 for player 2. Empty cells are represented as 0.
class Game:
    """
    Tic-Tac-Toe game environment for RL agents.
    Board is a 3x3 numpy array. Player 1 uses 1, Player 2 uses -1, empty is 0.
    """
    def __init__(self):
        """Initialize an empty 3x3 board."""
        self._state :np.ndarray = np.zeros((3, 3))

    def play(self, player: int, action: Tuple[int, int]) -> bool:
        """
        Attempt to play a move for the given player at the specified action (row, col).
        Returns True if the move is valid and applied, False otherwise.
        """
        # Input validation
        if not isinstance(player, int) or player not in [1, -1]:
            return False
        if (not isinstance(action, tuple) or len(action) != 2 or
            not all(isinstance(x, int) and 0 <= x < 3 for x in action)):
            return False

        # Check for legal move
        if self._state[action[0]][action[1]] != 0:
            return False

        # Update the _state based on the player's action
        self._state[action[0]][action[1]] = player

        return True

    def get_state(self) -> np.ndarray:
        """Return a copy of the current board state as a numpy array."""
        return self._state.copy()

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Return a list of all valid (row, col) actions for the current board state."""
        return [(i, j) for i in range(3) for j in range(3) if self._state[i][j] == 0]

    def check_win(self) -> bool:
        """Check if either player has won the game. Returns True if there is a winner."""
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
        """Check if the game is a draw. Returns True if there are no empty cells and no winner."""
        return all(cell != 0 for row in self._state for cell in row) and not self.check_win()
