"""
Party module for orchestrating a game session between two RL agents or players.
Handles turn alternation, reward assignment, and win/draw detection.
"""
from reinforcement_learning.game.Game import Game
from reinforcement_learning.players.Player import Player
from typing import Optional


class Party:
    """
    Manages a game session between two players (agents or humans).
    Alternates turns, handles rewards, and tracks the winner.
    """
    def __init__(self, player1: Player, player2: Player) -> None:
        """
        Initialize a Party with two players.
        Args:
            player1: First player (Player instance)
            player2: Second player (Player instance)
        """
        if not isinstance(player1, Player) or not isinstance(player2, Player):
            raise ValueError("Both player1 and player2 must be instances of Player.")
        self.game = Game()
        self.player1: Player = player1
        self.player2: Player = player2
        self.winner: Optional[Player] = None

    def play(self) -> None:
        """
        Run a full game session, alternating turns between players.
        Handles move selection, reward assignment, and win/draw detection.
        """
        current_player = self.player1
        other_player = self.player2

        while not self.game.check_win() and not self.game.check_draw():
            state = self.game.get_state()
            valid_actions = self.game.get_valid_actions()

            action = current_player.get_action(self.game.get_state(), valid_actions)
            if not self.game.play(current_player.token, action):
                raise ValueError(f"Invalid action {action} by player with token {current_player.token}")

            next_state = self.game.get_state()

            if self.game.check_win():
                current_player.on_action_taken(state, action, next_state, reward=1, done=True)
                other_player.on_episode_end(final_reward=-1)
                self.winner = current_player
                break

            if self.game.check_draw():
                current_player.on_action_taken(state, action, next_state, reward=0.5, done=True)
                other_player.on_episode_end(final_reward=0)
                break

            # Opponent plays next
            opp_action = other_player.get_action(
                self.game.get_state(),
                self.game.get_valid_actions()
            )
            self.game.play(other_player.token, opp_action)

            if self.game.check_win():
                current_player.on_action_taken(state, action, self.game.get_state(), reward=-1, done=True)
                other_player.on_episode_end(final_reward=1)
                self.winner = other_player
                break

            # Partie continue, no winner yet
            current_player.on_action_taken(
                state,
                action,
                self.game.get_state(),
                reward=0,
                done=False
            )

            current_player, other_player = other_player, current_player

    def reset_game(self) -> None:
        """
        Reset the game to play a new session.
        Reinitializes the game state and clears the winner.
        """
        self.game = Game()
        self.winner = None
