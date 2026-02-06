import numpy as np
from reinforcement_learning.game.Game import Game

def test_play():
    game = Game()
    # Test initial state
    assert np.array_equal(game.state, np.zeros((3, 3)))

    # Test legal move
    assert game.play(1, (0, 0)) == True
    assert game.state[0][0] == 1

    # Test illegal move
    assert game.play(-1, (0, 0)) == False
    assert game.state[0][0] == 1

    # Test out of bounds move
    assert game.play(1, (3, 3)) == False
    assert game.play(1, (-1, -1)) == False

def test_win():
    game = Game()
    # Test initial state
    game.state = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    assert game.check_win() == False

    # Rows
    game.state = np.array([
        [1, 1, 1],
        [0, -1, 0],
        [0, -1, 0]
    ])
    assert game.check_win() == True

    game.state = np.array([
        [0, -1, 0],
        [1, 1, 1],
        [-1, 0, 0]
    ])
    assert game.check_win() == True

    game.state = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [-1, -1, -1]
    ])
    assert game.check_win() == True

    # Columns
    game.state = np.array([
        [1, 0, 0],
        [1, -1, 0],
        [1, 0, -1]
    ])
    assert game.check_win() == True

    game.state = np.array([
        [0, 1, 0],
        [0, 1, -1],
        [0, 1, 0]
    ])
    assert game.check_win() == True

    game.state = np.array([
        [0, 0, 1],
        [0, -1, 1],
        [0, 0, 1]
    ])
    assert game.check_win() == True

    # Diagonals
    game.state = np.array([
        [1, -1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    assert game.check_win() == True

    game.state = np.array([
        [0, 1, -1],
        [0, -1, 0],
        [-1, 0, 0]
    ])
    assert game.check_win() == True


if __name__ == "__main__":
    test_win()
