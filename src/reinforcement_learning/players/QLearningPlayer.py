import random
from typing import List, Tuple

import numpy as np

from reinforcement_learning.players.Player import Player


class QLearningPlayer(Player):
    def __init__(self, token: int, alpha=0.1, gamma=0.9, epsilon=0.2):
        super().__init__(token)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Q[(state_tuple, action)] -> float

    def _state_to_key(self, state: np.ndarray):
        return tuple(state.flatten())

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]):
        state_key = self._state_to_key(state)

        # exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # exploitation
        q_values = []
        for action in valid_actions:
            q_values.append(self.q_table.get((state_key, action), 0.0))

        max_q = max(q_values)
        best_actions = [
            a for a, q in zip(valid_actions, q_values) if q == max_q
        ]
        return random.choice(best_actions)

    def on_action_taken(self, state, action, next_state, reward, done):
        self.update(state, action, next_state, reward, done)

    def on_episode_end(self, final_reward):
        pass

    def update(self, state, action, next_state, reward, done):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        old_q = self.q_table.get((state_key, action), 0.0)

        if done:
            target = reward
        else:
            future_q = max(
                self.q_table.get((next_state_key, a), 0.0)
                for a in range(9)
            )
            target = reward + self.gamma * future_q

        self.q_table[(state_key, action)] = old_q + self.alpha * (target - old_q)
