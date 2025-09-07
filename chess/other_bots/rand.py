import chess

__version__ = "0.0"

from random import choice

__version__ = None

class Computer:
    
    def __init__(self, color):
        self.color = color

    def best_move(self, board: chess.Board, *, timeout: float=float('inf')) -> chess.Move | None:
        return choice(list(board.legal_moves)) if board.legal_moves else None