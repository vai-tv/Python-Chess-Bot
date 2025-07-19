import chess
import json
import sqlite3
import time


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

        self.timeout: float | None = None
        self.start_time: float | None = None

        self.init_db()

    ##################################################
    #                    DATABASES                   #
    ##################################################

    TRANSPOSITION_PATH = "bot/transposition_table.db"

    @classmethod
    def init_db(cls):
        """
        Initialize SQLite database for transposition table stored in file.

        Connects to the SQLite database, creates the table if it does not exist, and
        commits the changes.
        """
        
        # Initialize SQLite database for transposition table stored in file
        cls.conn = sqlite3.connect(cls.TRANSPOSITION_PATH)
        cls.cursor = cls.conn.cursor()
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transposition_table (
                zobrist_key TEXT PRIMARY KEY,
                score REAL,
                depth INTEGER
            )
        """)
        cls.conn.commit()

    def evaluate_from_db(self, board: chess.Board, depth: int = 0) -> float | None:
        """
        Evaluate the given board position from the transposition table in the database.

        Args:
            board: The chess board to evaluate.
            depth: The minimum depth to consider in the database.

        Returns:
            The score of the board position if it exists in the database, otherwise None.
        """
        
        # TODO: Implement zobrist
        zobrist_key = board.fen()
        self.cursor.execute("SELECT score FROM transposition_table WHERE zobrist_key = ? AND depth >= ?", (zobrist_key, depth,))
        row = self.cursor.fetchone()
        if row is not None:
            return row[0]
        return None

    def save_evaluation(self, board: chess.Board, score: float, depth: int) -> None:
        """
        Save an evaluation of a given board position to the transposition table in the database.

        Args:
            board: The chess board to evaluate.
            score: The score of the board position.
            depth: The depth at which the score was evaluated.

        Raises:
            sqlite3.Error: If there is an error saving the evaluation to the database.
        """
        
        # TODO: Implement zobrist
        zobrist_key = board.fen()
        try:
            # Check existing depth for the zobrist_key
            self.cursor.execute("SELECT depth FROM transposition_table WHERE zobrist_key = ?", (zobrist_key,))
            row = self.cursor.fetchone()
            if row is None:
                # No existing entry, insert new
                self.cursor.execute("INSERT INTO transposition_table (zobrist_key, score, depth) VALUES (?, ?, ?)", (zobrist_key, score, depth,))
                # self.conn.commit()
            else:
                existing_depth = row[0]
                if depth > existing_depth:
                    # Update only if new depth is higher
                    self.cursor.execute("UPDATE transposition_table SET score = ?, depth = ? WHERE zobrist_key = ?", (score, depth, zobrist_key))
                    # self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving evaluation to DB: {e}")

    ##################################################
    #                   HEURISTICS                   #
    ##################################################

    def _score_weak_heuristic(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        """
        Evaluate and score each legal move from the current board position using the Minimax algorithm 
        without heuristic sorting or elimination.

        :param board: The current state of the chess board.
        :type board: chess.Board
        :return: A list of tuples containing legal moves and their corresponding scores.
        :rtype: list[tuple[chess.Move, float]]
        """

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
        """
        Find the index of the 'turning point' in sorted scores using the elbow method.

        The elbow method is a technique to find the cutoff point in a sorted list of numbers.
        It works by finding the point with the maximum distance to a line going through the
        first and last point of the list.

        The method takes a sorted list of scores as input and returns the index of the
        turning point. If the list is empty or contains only one element, the method
        returns -1 or 0 respectively.

        :param scores: A sorted list of scores
        :return: The index of the turning point
        """

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

        # Try to get score from DB
        cached_score = self.evaluate_from_db(board)
        if cached_score is not None:
            return cached_score

        # Game over
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

            attack_bonus = 0

            # Reward squares covered
            attacked_squares: list[chess.Square] = []
            for square, piece in board.piece_map().items():
                if piece.color == color:
                    attacked_squares.extend(list(board.attacks(square)))
            
            for square in set(attacked_squares):
                attack_bonus += attacked_squares.count(square) ** 0.8

                # Give bonuses for attacking high value pieces, give bonuses to defending low value pieces
                target = board.piece_at(square)
                if target is None:
                    continue
                if target.color == color:
                    attack_bonus += self.MATERIAL[target.piece_type] ** 1.5 * 0.5
                else:
                    attack_bonus += self.MATERIAL[target.piece_type] * 0.75 * 0.5

            score += attack_bonus ** 1.5 * 0.05

            return score
        
        stage = self.get_game_stage(board)
        score = 0

        # Material
        for square, piece in board.piece_map().items():
            score += self.MATERIAL[piece.piece_type] * (1 if piece.color == chess.WHITE else -1) ** 2 * 10
        
        # Heatmap
        for square, piece in board.piece_map().items():
            piece_symbol = piece.symbol().upper()

            # Convert square index to rank and file
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Heatmaps are incorrectly oriented for python chess; flip them
            if piece.color == chess.WHITE:
                rank = 7 - rank
            score += self.HEATMAP[stage][piece_symbol][rank][file] * (1 if piece.color == chess.WHITE else -1)

        # Player evaluation
        score += evaluate_player(chess.WHITE)
        score -= evaluate_player(chess.BLACK)

        # Save evaluation to DB
        self.save_evaluation(board, score, 0)

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
            return any(score == self.BEST_SCORE for _, score in move_score_map) or self.is_timeup()

        self.start_time = time.time()
        self.timeout = timeout

        depth = 1
        best_move = None

        moves = list(board.legal_moves)  # Convert generator to list for membership checks

        move_score_map: list[tuple[chess.Move, float]] = [] # js so pylance shuts up about it being unbound :sob:

        while not self.is_timeup():

            print(f"""
            DEPTH {depth}

            """)

            # Gradually filter out based on the previous scores
            if depth > 1:

                move_score_map.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)

                turning_point = self._turning_point([score for _, score in move_score_map])
                moves = [move for move, _ in move_score_map[:turning_point]]
                print(len(moves),"left")

            moves_set = set(moves)  # Use a set for efficient membership checking

            move_score_map: list[tuple[chess.Move, float]] = []

            for move in moves:
                board.push(move)

                # vv CURRENTLY UNREACHABLE vv
                if move not in moves_set:
                    score = self.evaluate_from_db(board, depth)
                    if score is None:
                        board.pop()
                        continue
                else:
                    score = self.minimax(board, depth, float('-inf'), float('inf'))
                # ^^ CURRENTLY UNREACHABLE ^^

                print(f"{move}: {score}")
                board.pop()
                
                move_score_map.append((move, score))

                self.save_evaluation(board, score, depth)

                if _should_terminate(move_score_map):
                    break
            
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

            self.conn.commit()

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

    def is_timeup(self) -> bool:
        if self.timeout is None or self.start_time is None:
            return False
        return time.time() - self.start_time > self.timeout

def main():

    FEN = "r3qrk1/2bn1ppp/p1p5/1p6/3P4/P1NB2QR/1PP2PPP/R5K1 w - - 0 1"
    FEN = chess.STARTING_BOARD_FEN

    board = chess.Board(FEN)
    computer = Computer(chess.WHITE)

    while not board.is_game_over():
        move = computer.best_move(board, timeout=10)
        if move is None:
            break
        board.push(move)
        print("Move:", move)
        print(board,"\n\n")

if __name__ == "__main__":
    main()
