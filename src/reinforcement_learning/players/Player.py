import numpy as np

from typing import List, Tuple


class Player:
    def __init__(self, token: int):
        if not isinstance(token, int):
            raise ValueError("Token must be an integer. Use 1 for player 1 and -1 for player 2.")
        self.token = token
        self.is_bot = True

    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int]]) -> Tuple[int]:
        # This method should be overridden by subclasses to implement specific strategies.
        raise NotImplementedError("This method should be overridden by subclasses.")

    # hooks RL (default : no-op)
    def on_action_taken(self, state, action, next_state, reward, done):
        pass

    def on_episode_end(self, final_reward):
        pass