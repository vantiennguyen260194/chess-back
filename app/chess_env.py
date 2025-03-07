import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board = chess.Board()
        return self.board.fen()

    def is_done(self):
        return self.board.is_game_over()

    def get_valid_moves(self):
        return list(self.board.legal_moves)

    def step(self, move):
        self.board.push(move)
        reward = self.get_reward()
        done = self.is_done()
        return self.board.fen(), reward, done

    def get_reward(self):
        if self.board.is_checkmate():
            return -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        elif self.board.is_check():
            return 1
        else: 
            return 0

