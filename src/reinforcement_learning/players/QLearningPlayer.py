"""
QLearningPlayer implements a Q-learning agent for Tic-Tac-Toe.
Maintains a Q-table and uses epsilon-greedy exploration.
"""
import random
from typing import List, Tuple, Any

import numpy as np

from reinforcement_learning.players.Player import Player


class QLearningPlayer(Player):
    """
    Q-learning agent for Tic-Tac-Toe.
    Uses a Q-table for state-action values and epsilon-greedy exploration.
    """

    def __init__(self, token: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2) -> None:
        """
        Initialize a QLearningPlayer.
        Args:
            token: 1 for player 1, -1 for player 2
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        if not (0 <= epsilon <= 1):
            raise ValueError("epsilon must be between 0 and 1.")
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1].")
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be in (0, 1].")
        super().__init__(token)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Q[(state_tuple, action)] -> float

    def _state_to_key(self, state: np.ndarray) -> Tuple:
        """
        Convert a board state to a hashable key for the Q-table.
        Args:
            state: Board state (numpy array)
        Returns:
            Tuple representing the flattened board state
        """
        return tuple(state.flatten())

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select an action using epsilon-greedy policy.
        Args:
            state: Current board state
            valid_actions: List of valid actions
        Returns:
            Selected action (row, col)
        """
        state_key = self._state_to_key(state)

        if not valid_actions:
            raise ValueError("No valid actions available.")

        # exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # exploitation
        q_values = [self.q_table.get((state_key, action), 0.0) for action in valid_actions]

        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def on_action_taken(self, state: Any, action: Any, next_state: Any, reward: float, done: bool) -> None:
        """
        Update Q-table after an action is taken.
        Args:
            state: Previous state
            action: Action taken
            next_state: Resulting state
            reward: Reward received
            done: Whether the episode is finished
        """
        self.update(state, action, next_state, reward, done)

    def on_episode_end(self, final_reward: float) -> None:
        pass

    def update(self, state: Any, action: Any, next_state: Any, reward: float, done: bool) -> None:
        """
        Q-learning update rule for the Q-table.
        """
        state = self.token * state  # agent sees the board from its perspective
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        old_q = self.q_table.get((state_key, action), 0.0)

        if done:
            target = reward
        else:
            # Estimate of optimal future value
            next_qs = [self.q_table.get((next_state_key, a), 0.0) for a in range(9)]
            target = reward + self.gamma * max(next_qs, default=0.0)

        self.q_table[(state_key, action)] = old_q + self.alpha * (target - old_q)

    def deactivate_learning(self) -> None:
        """Stop exploration by setting epsilon to 0."""
        self.epsilon = 0.0
