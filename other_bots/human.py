import chess

class Human:
    def __init__(self, color):
        self.color = color

    def best_move(self, board: chess.Board, *, timeout: float=float('inf')) -> chess.Move:
        move = input("Enter your move: ")

        while True:
            try:
                if chess.Move.from_uci(move) in board.legal_moves:
                    return chess.Move.from_uci(move)
                else:
                    print("Invalid move: Illegal move.")
            except chess.InvalidMoveError:
                print("Invalid move: Not a UCI string.")