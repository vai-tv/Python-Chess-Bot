import chess

__version__ = None

class Computer:
    
    def __init__(self, color):
        self.color = color

    def best_move(self, board: chess.Board, *, timeout: float=float('inf')) -> chess.Move:
        while True:
            move = input("Enter your move in SAN notation: ")
            try:
                move_obj = board.parse_san(move)
                if move_obj in board.legal_moves:
                    return move_obj
                else:
                    print("Invalid move: Illegal move.")
            except ValueError:
                print("Invalid move: Not a valid SAN string.")
