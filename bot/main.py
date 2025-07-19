import chess
import json
import time

from typing import Literal

class Computer:

    p = chess.Piece.from_symbol('P')

    MATERIAL: dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    HEATMAP_PATH = "bot/heatmap.json"
    HEATMAP = json.load(open(HEATMAP_PATH))

    def __init__(self, color: chess.Color):
        self.color = color
        self.BEST_SCORE = float('inf') if color == chess.WHITE else float('-inf')
        self.WORST_SCORE = float('-inf') if color == chess.WHITE else float('inf')
        self.MAXMIN = max if color == chess.WHITE else min

    ##################################################
    #                   HEURISTICS                   #
    ##################################################

    def _score_weak_heuristic(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        weak_depth = 0
        move_score_map: list[tuple[chess.Move, float]] = []

        for move in board.legal_moves:
            board.push(move)
            score = self.minimax(board, weak_depth, float('-inf'), float('inf'), heuristic_sort=False, heuristic_eliminate=False)
            board.pop()
            move_score_map.append((move, score))
        
        return move_score_map

    def weak_heuristic_moves(self, board: chess.Board) -> list[chess.Move]:
        """
        Generate a list of moves in weak heuristic order.

        :param board: The current state of the board
        :type board: chess.Board
        :return: A list of moves in weak heuristic order
        :rtype: list[chess.Move]
        """

        move_score_map = self._score_weak_heuristic(board)
        
        # Sort moves by score
        sorted_moves = sorted(move_score_map, key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        return [move for move, _ in sorted_moves]
    
    def _turning_point(self, scores: list[float]) -> int:
        # Elbow method to find cutoff index in sorted scores
        if not scores:
            return -1
        if len(scores) == 1:
            return 0
    
        min_kept = len(scores) // 3
        max_kept = (len(scores) * 2) // 3

        # Coordinates of points: (index, score)
        points = [(i, score) for i, score in enumerate(scores)]

        # Line between first and last point
        start = points[0]
        end = points[-1]

        def distance(point, start, end):
            # Calculate perpendicular distance from point to line (start-end)
            x0, y0 = point
            x1, y1 = start
            x2, y2 = end
            numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
            denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
            if denominator == 0:
                return 0
            return numerator / denominator

        # Find point with max distance to line
        max_dist = -1
        max_index = 0
        for i in range(1, len(points) - 1):
            dist = distance(points[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_index = i

        return max(min(max_kept, max_kept), min_kept)

    def select_wh_moves(self, board: chess.Board) -> list[chess.Move]:
        """
        Select some moves worth exploring based on a weak heuristic.

        :param board: The current state of the board
        :type board: chess.Board
        :return: A list of moves in weak heuristic order
        :rtype: list[chess.Move]
        """

        move_score_map = self._score_weak_heuristic(board)
        
        # Sort moves by score
        sorted_moves = sorted(move_score_map, key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        
        # Decide turning point (elbow function)
        scores = [score for _, score in sorted_moves]
        cutoff_index = self._turning_point(scores)
        selected_moves = [move for move, _ in sorted_moves[:cutoff_index+1]]

        return selected_moves

    ##################################################
    #                   EVALUATION                   #
    ##################################################

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, *, heuristic_sort: bool = True, heuristic_eliminate: bool = True) -> float:
        """
        Evaluate the best move to make using the Minimax algorithm.

        :param board: The current state of the board
        :type board: chess.Board
        :param depth: The number of moves to look ahead
        :type depth: int
        :param alpha: The best score possible for the maximizing player
        :type alpha: float
        :param beta: The best score possible for the minimizing player
        :type beta: float
        :param heuristic_sort: Whether to sort moves by heuristic score
        :type heuristic_sort: bool
        :param heuristic_eliminate: Whether to eliminate moves with low heuristic scores
        :type heuristic_eliminate: bool
        :return: The best score possible for the maximizing player
        :rtype: float
        """

        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        is_maximizing = board.turn == chess.WHITE
        best_score = float('-inf') if is_maximizing else float('inf')

        if heuristic_eliminate:
            legals = self.select_wh_moves(board)
        elif heuristic_sort:
            legals = self.weak_heuristic_moves(board)
        else:
            legals = list(board.legal_moves)

        for move in legals:
            board.push(move)
            score = self.minimax(board, depth - 1, alpha, beta, heuristic_sort=heuristic_sort)
            board.pop()

            if is_maximizing:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        return best_score

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate the board state and return a score.

        :param board: The current state of the board
        :type board: chess.Board
        :return: A numerical evaluation of the board state
        :rtype: float
        """

        # Game-over
        if board.is_game_over():
            if board.result() == "1-0":
                return float('inf')
            elif board.result() == "0-1":
                return float('-inf')
            else:
                print("DRAW")
                return 0

        def evaluate_player(color: chess.Color) -> float:
            score = 0

            return score
        
        stage = self.get_game_stage(board)
        score = 0

        # Material
        for square, piece in board.piece_map().items():
            score += self.MATERIAL[piece.piece_type] * (1 if piece.color == chess.WHITE else -1) ** 2 * 10
        
        for square, piece in board.piece_map().items():
            piece_symbol = piece.symbol().upper()

            # Convert square index to rank and file
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Heatmaps are incorrectly oriented for python chess; flip them
            if piece.color == chess.WHITE:
                rank = 7 - rank
            score += self.HEATMAP[stage][piece_symbol][rank][file] * (1 if piece.color == chess.WHITE else -1)

        score += evaluate_player(chess.WHITE)
        score -= evaluate_player(chess.BLACK)
        return score

    def best_move(self, board: chess.Board, timeout: float=float('inf')) -> chess.Move | None:
        
        """
        Determine and return the best move for the computer player using the Minimax algorithm.

        :param board: The current state of the board
        :type board: chess.Board
        :param timeout: The maximum allowed time to find the best move
        :type timeout: float
        :return: The best legal move for the computer player, or None if no move is possible
        :rtype: chess.Move | None
        """

        def _should_terminate(move_score_map: list[tuple[chess.Move, float]]) -> bool:
            return any(score == self.BEST_SCORE for _, score in move_score_map) or time.time() - start_time > timeout

        start_time = time.time()

        depth = 1
        best_move = None

        legals = board.legal_moves

        while time.time() - start_time < timeout:

            print(f"Depth: {depth}")

            move_score_map: list[tuple[chess.Move, float]] = []

            # Gradually filter out based on the previous scores
            if depth > 1:
                turning_point = self._turning_point([score for _, score in move_score_map])
                print(turning_point)
                legals = [move for move, _ in move_score_map[:turning_point]]
                print(len(legals),"left")

            for move in legals:
                board.push(move)
                score = self.minimax(board, depth, float('-inf'), float('inf'))
                board.pop()
                
                move_score_map.append((move, score))
            
            # Terminate early if an immediate win is found
            if _should_terminate(move_score_map):
                if board.turn == chess.WHITE:
                    best_move = max(move_score_map, key=lambda x: x[1])[0]
                else:
                    best_move = min(move_score_map, key=lambda x: x[1])[0]
                break

            if board.turn == chess.WHITE:
                best_move = max(move_score_map, key=lambda x: x[1])[0]
            else:
                best_move = min(move_score_map, key=lambda x: x[1])[0]

            print("Best:", best_move)

            depth += 1

        return best_move

    ##################################################
    #                     EXTRAS                     #
    ##################################################

    def get_game_stage(self, board: chess.Board) -> str:
        """Return the current stage of the game."""

        num_pieces = len(board.piece_map().values())
        if num_pieces > 20:
            return "early"
        elif num_pieces > 10:
            return "middle"
        else:
            return "late"

def main():

    FEN = "r3qrk1/2bn1ppp/p1p5/1p6/3P4/P1NB2QR/1PP2PPP/R5K1 w - - 0 1"

    board = chess.Board(FEN)
    computer = Computer(chess.WHITE)

    while not board.is_game_over():
        move = computer.best_move(board)
        if move is None:
            break
        board.push(move)
        print(move)
        print(board,"\n\n")

if __name__ == "__main__":
    main()
