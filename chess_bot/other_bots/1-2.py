import chess
import json
import os
import sqlite3
import time

__version__ = '1.2'

class Computer:

    p = chess.Piece.from_symbol('P')

    MATERIAL: dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 25
    }

    HEATMAP_PATH = "chess_bot/global-assets/heatmap.json"
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

    TRANSPOSITION_PATH = f"chess_bot/tables/{__version__}_transposition.db"

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
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS winning_moves (
                position_fen TEXT PRIMARY KEY,
                move_uci TEXT
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
    
        # min_kept = max(len(scores) // 2, 1)
        min_kept = 1
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
        max_index = min_kept
        for i in range(min_kept, max_kept):
            dist = distance(points[i], start, end)
            # Discourage turning points that are too low by adding a small penalty
            penalty = 0
            if i < len(scores) // 4:
                penalty = 0.1  # small penalty to discourage too low turning points
            dist -= penalty
            if dist > max_dist:
                max_dist = dist
                max_index = i

        return max_index + 1

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

        def save_winning_move_local(board_before_move: chess.Board, move: chess.Move) -> None:
            fen = board_before_move.fen()
            move_uci = move.uci()
            try:
                self.cursor.execute("INSERT OR REPLACE INTO winning_moves (position_fen, move_uci) VALUES (?, ?)", (fen, move_uci))
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error saving winning move to DB: {e}")

        if depth == 0 or board.is_game_over() or self.is_timeup():
            return self.evaluate(board)

        is_maximizing = board.turn == chess.WHITE
        best_score = float('-inf') if is_maximizing else float('inf')
        best_move = None

        if heuristic_eliminate:
            legals = self.select_wh_moves(board)
        elif heuristic_sort:
            legals = self.weak_heuristic_moves(board)
        else:
            legals = list(board.legal_moves)

        for move in legals:
            board.push(move)
            score = self.minimax(board, depth - 1, alpha, beta, heuristic_sort=heuristic_sort, heuristic_eliminate=heuristic_eliminate)
            board.pop()

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        # If this position leads to a winning score, save the winning move
        if ((is_maximizing and best_score == float('inf')) or (not is_maximizing and best_score == float('-inf'))) and best_move is not None:
            save_winning_move_local(board, best_move)

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
                return 0
        
        stage = self.get_game_stage(board)

        def evaluate_player(color: chess.Color) -> float:
            # Reward squares covered
            attack_bonus = 0
            attacked_squares: list[chess.Square] = []
            for square, piece in board.piece_map().items():
                if piece.color == color:
                    attacked_squares.extend(list(board.attacks(square)))
            
            for square in set(attacked_squares):
                attack_bonus += attacked_squares.count(square) ** 0.8

                # Reward centre control
                if square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                    attack_bonus += 2.5 * aggression[color]
                # Reward control of enemy half
                if color == chess.WHITE and chess.square_rank(square) > 3:
                    attack_bonus += 1.5 * aggression[color]
                elif color == chess.BLACK and chess.square_rank(square) < 4:
                    attack_bonus += 1.5 * aggression[color]

                # Give bonuses for attacking high value pieces, give bonuses to defending low value pieces
                target = board.piece_at(square)
                if target is None:
                    continue
                attack_bonus += self.MATERIAL[target.piece_type] ** 1.5
            
            # Reward pieces per square control
            control_bonus = 0
            for square in board.piece_map().keys():
                attacks = board.attacks(square)
                control_bonus += len(attacks) ** 0.75

                if square not in [chess.E4, chess.E5, chess.D4, chess.D5]:
                    continue
                piece = board.piece_at(square)
                if piece is None or piece.color != color:
                    continue
                control_bonus += self.MATERIAL[piece.piece_type]
                if piece == chess.PAWN:
                    control_bonus += 2

            # Material
            material_score = material[color]

            # Heatmap
            heatmap_score: float = 0
            for square, piece in board.piece_map().items():
                if piece.color != color:
                    continue
                piece_symbol = piece.symbol().upper()

                # Convert square index to rank and file
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                # Heatmaps are incorrectly oriented for python chess; flip them
                if piece.color == chess.WHITE:
                    rank = 7 - rank
                heatmap_score += self.HEATMAP[stage][piece_symbol][rank][file]# * (self.MATERIAL[piece.piece_type] ** 0.5)
            
            # Punish low number of legal moves
            if color == board.turn:
                legal_moves = board.legal_moves
            else:
                old_turn = board.turn
                board.turn = color
                legal_moves = board.legal_moves
                board.turn = old_turn

            # Bonus for minor piece development
            minor_piece_development_bonus = 0
            starting_squares = {
                chess.WHITE: {
                    chess.KNIGHT: [chess.B1, chess.G1],
                    chess.BISHOP: [chess.C1, chess.F1]
                },
                chess.BLACK: {
                    chess.KNIGHT: [chess.B8, chess.G8],
                    chess.BISHOP: [chess.C8, chess.F8]
                }
            }
            for square, piece in board.piece_map().items():
                if piece.color == color and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if square not in starting_squares[color][piece.piece_type]:
                        minor_piece_development_bonus += 1.5  # Bonus for developed minor piece


            # Might be removed later : Penalty for early queen activity
            queen_penalty = 0
            queen_squares = [square for square, piece in board.piece_map().items() if piece.color == color and piece.piece_type == chess.QUEEN]
            for q_square in queen_squares:
                rank = chess.square_rank(q_square)
                if stage == 'early':
                    # Penalize queen on ranks 3,4,5 (too early and too advanced)
                    if (color == chess.WHITE and rank <= 4) or (color == chess.BLACK and rank >= 3):
                        queen_penalty += 2.0
                elif stage == 'middle':
                    # Smaller penalty in middle game for queen too advanced
                    if (color == chess.WHITE and rank <= 3) or (color == chess.BLACK and rank >= 4):
                        queen_penalty += 1.0
            
            # King safety penalties
            king_penalty = 0
            # Since python-chess does not have has_castled method, check castling rights and castling status differently
            # We can check if the castling rights are lost but the king is still on the original square, meaning no castling yet
            king_start_square = chess.E1 if color == chess.WHITE else chess.E8
            king = board.king(color)
            has_moved = king != king_start_square
            if stage != 'late' and (not board.has_castling_rights(color)) and has_moved:
                king_penalty += 10.0
            if king is None:
                return float('-inf') if color == chess.WHITE else float('inf')
            king_moves = list(board.attacks(king))
            king_penalty -= len(king_moves) ** 0.5 # Penalise less for more king moves

            score = 0            
            score -= king_penalty * 2
            score -= queen_penalty / aggression[color]
            score -= (5 / len(list(legal_moves))) ** 1.5 * (aggression[not color] ** 2)
            score += (material_score ** 2 * 10) * aggression[color]
            score += (attack_bonus ** 1.5 * 0.1) * aggression[color]
            score += (heatmap_score * 15)
            score += (control_bonus ** 1.5) * 0.35 * aggression[color]
            score += minor_piece_development_bonus * aggression[color]

            return score
        
        # Material
        material = {chess.WHITE: 0, chess.BLACK: 0}
        for _, piece in board.piece_map().items():
            if piece.color == chess.WHITE:
                material[chess.WHITE] += self.MATERIAL[piece.piece_type]
            else:
                material[chess.BLACK] += self.MATERIAL[piece.piece_type]
        
        # Aggression
        aggression = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        for color in [chess.WHITE, chess.BLACK]:
            aggression[color] = min(material[color] / (2 * material[not color]), 1.5) ** 2
            aggression[color] *= 0.5 if stage == 'early' else 1.25 if stage == 'middle' else 1
        
        # Player evaluation
        score = 0
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
            return any(score == self.BEST_SCORE for _, score in move_score_map)

        def get_stored_winning_move(board: chess.Board) -> chess.Move | None:
            fen = board.fen()
            self.cursor.execute("SELECT move_uci FROM winning_moves WHERE position_fen = ?", (fen,))
            row = self.cursor.fetchone()
            if row is not None:
                try:
                    return chess.Move.from_uci(row[0])
                except:
                    return None
            return None

        def save_winning_move(board_before_move: chess.Board, move: chess.Move) -> None:
            fen = board_before_move.fen()
            move_uci = move.uci()
            try:
                self.cursor.execute("INSERT OR REPLACE INTO winning_moves (position_fen, move_uci) VALUES (?, ?)", (fen, move_uci))
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error saving winning move to DB: {e}")

        print(f"{"white" if board.turn == chess.WHITE else "black"} move")

        self.conn = sqlite3.connect(self.TRANSPOSITION_PATH)
        self.cursor = self.conn.cursor()

        self.start_time = time.time()
        self.timeout = timeout

        # Check if there is a stored winning move for the current position
        stored_move = get_stored_winning_move(board)
        if stored_move is not None and stored_move in board.legal_moves:
            print("Using stored winning move")
            # self.conn.close()  # Removed to prevent closing connection prematurely
            return stored_move

        depth = 1
        best_move = None
        save = True

        maxmin = max if board.turn == chess.WHITE else min

        moves = list(board.legal_moves)  # Convert generator to list for membership checks

        move_score_map: list[tuple[chess.Move, float]] = [(move, 0) for move in moves] # Initialize once before loop

        # Keep track of best moves and scores from last few fully completed depths
        best_moves_history: list[list[tuple[chess.Move, float]]] = []

        while not self.is_timeup():

            print(f"""DEPTH {depth}: """,end='\t')

            move_score_map.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
            moves = [move for move, _ in move_score_map]

            # Gradually filter out based on the previous scores
            if depth > 3:
                turning_point = self._turning_point([score for _, score in move_score_map])
                move_score_map = move_score_map[:turning_point]
                moves = moves[:turning_point]
                print(len(moves),"moves to look at:",[board.san(m) for m in moves])

            # If only one move left, return it
            if len(moves) == 1:
                return moves[0]

            moves_set = set(moves)  # Use a set for efficient membership checking

            # Create a dictionary for quick lookup and update of scores
            move_score_dict = dict(move_score_map)

            for move in moves:
                board.push(move)

                # vv CURRENTLY UNREACHABLE vv
                if move not in moves_set:
                    score = self.evaluate_from_db(board, depth)
                    if score is None:
                        board.pop()
                        continue
                # ^^ CURRENTLY UNREACHABLE ^^
                else:
                    score = self.evaluate_from_db(board, depth)
                    if score is not None:
                        board.pop()
                        move_score_dict[move] = score
                        continue
                    score = self.minimax(board, depth, float('-inf'), float('inf'), heuristic_eliminate=False)

                board.pop()
                print(f"{board.san(move)} : {score:.2f}",end='\t',flush=True)
                
                move_score_dict[move] = score

                self.save_evaluation(board, score, depth)

                if _should_terminate(list(move_score_dict.items())):
                    print("TERMINATED EARLY")
                    break
                if self.is_timeup():
                    print("TIMEUP")
                    save = False
                    break
            
            print()

            # Update move_score_map from the dictionary for next iteration
            move_score_map = list(move_score_dict.items())

            # Save the current depth's move scores to history if search completed fully
            if not self.is_timeup() and not _should_terminate(move_score_map):
                best_moves_history.append(move_score_map)
                # Keep only the last 3 depths
                if len(best_moves_history) > 3:
                    best_moves_history.pop(0)

            # Terminate early if an immediate win is found
            if _should_terminate(move_score_map) or all(score == self.WORST_SCORE for _, score in move_score_map):
                # Choose best move from current depth
                best_move = maxmin(move_score_map, key=lambda x: x[1])[0]

                # Save the winning move and the position before the move
                save_winning_move(board, best_move)
                break

            # Choose best move from the last fully completed depths history
            if best_moves_history:
                # Aggregate scores for moves appearing in history
                aggregate_scores = {}
                for depth_scores in best_moves_history:
                    for move, score in depth_scores:
                        aggregate_scores[move] = aggregate_scores.get(move, 0) + score
                # Average scores
                for move in aggregate_scores:
                    aggregate_scores[move] /= len(best_moves_history)

                # If timeout occurred, compare current depth scores with previous depth scores
                if self.is_timeup() and len(best_moves_history) > 1:
                    print("Timeout occurred, comparing with previous depth scores")
                    prev_scores = {move: score for move, score in best_moves_history[-2]}
                    # Check if all current moves are worse than previous best move
                    current_best_score = max(aggregate_scores.values()) if board.turn == chess.WHITE else min(aggregate_scores.values())
                    prev_best_score = max(prev_scores.values()) if board.turn == chess.WHITE else min(prev_scores.values())
                    if (board.turn == chess.WHITE and current_best_score < prev_best_score) or (board.turn == chess.BLACK and current_best_score > prev_best_score):
                        # Revert to previous depth's best move
                        best_move = maxmin(prev_scores.items(), key=lambda x: x[1])[0]
                    else:
                        best_move = maxmin(aggregate_scores.items(), key=lambda x: x[1])[0]
                else:
                    best_move = maxmin(aggregate_scores.items(), key=lambda x: x[1])[0]
            else:
                # Fallback to current depth best move
                best_move = maxmin(move_score_map, key=lambda x: x[1])[0]

            print("BEST:",board.san(best_move))

            depth += 1

            if save:
                self.conn.commit()

        self.conn.close()

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

    # FEN = "r1bq1rk1/ppp2ppp/2n5/3n2N1/3P4/2PB4/P4PP1/R2QK2R w - - 0 1"
    FEN = chess.STARTING_BOARD_FEN

    board = chess.Board(FEN)
    players = [Computer(board.turn), Computer(not board.turn)]

    while not board.is_game_over():
        print(board,"\n\n")
        player = players[0] if board.turn == chess.WHITE else players[1]
        move = player.best_move(board, timeout=20)
        if move is None:
            break
        board.push(move)
        print("Move:", move)
    print(board)
    print("GAME OVER!")

if __name__ == "__main__":
    main()
