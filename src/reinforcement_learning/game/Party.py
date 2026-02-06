from reinforcement_learning.game.Game import Game
from reinforcement_learning.Players.Player import Player

class Party:
    def __init__(self, player1, player2):
        self.game = Game()
        self.player1: Player = player1
        self.player2: Player = player2

    def play(self):
        previous_player = 0
        current_player = self.player1
        while not self.game.check_win() and not self.game.check_draw():
            action = current_player.get_action(self.game._state)
            while not self.game.play(current_player.token, action):
                print("Invalid move. Please try again.")
                action = current_player.get_action(self.game._state)

            previous_player = current_player
            current_player = self.player2 if current_player == self.player1 else self.player1

        if self.game.check_win():
            print(f"Player {previous_player.token} wins!")
        else:
            print("It's a draw!")

        print(self.game._state)
