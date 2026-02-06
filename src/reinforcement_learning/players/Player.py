class Player:
    def __init__(self, token: int):
        if not isinstance(token, int):
            raise ValueError("Token must be an integer. Use 1 for player 1 and -1 for player 2.")
        self.token = token
        self.is_bot = True

    def get_action(self, state)-> tuple:
        # This method should be overridden by subclasses to implement specific strategies.
        raise NotImplementedError("This method should be overridden by subclasses.")