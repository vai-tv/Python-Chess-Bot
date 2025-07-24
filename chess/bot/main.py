import chess
import json
import random as rnd
import requests
import sqlite3
import time
import urllib.parse

__version__ = '1.1.6'

class Computer:

    def __init__(self, color: chess.Color):
        self.color = color
        self.BEST_SCORE = float('inf') if color == chess.WHITE else float('-inf')
        self.WORST_SCORE = float('-inf') if color == chess.WHITE else float('inf')
        self.MAXMIN = max if color == chess.WHITE else min

        self.timeout: float | None = None
        self.start_time: float | None = None

        # Initialize killer moves dictionary: depth -> list of killer moves
        self.killer_moves: dict[int, list[chess.Move]] = {}

        self.init_db()

    ##################################################
    #                    DATABASES                   #
    ##################################################

    TRANSPOSITION_PATH = f"chess/tables/{__version__}_transposition.db"

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
    #            OPENING BOOKS AND SYGYZY            #
    ##################################################

    SYGYZY_URL = "https://tablebase.lichess.ovh/standard?fen="
    OPENING_URL = "https://explorer.lichess.ovh/master?fen="

    OPENING_LEAVE_CHANCE = 0.01  # Initial chance to leave the opening book, increases over time

    def sygyzy_query(self, board: chess.Board) -> dict:
        """
        Query the Syzygy tablebase server for the given board position.

        Args:
            board (chess.Board): The chess board position to query.

        Returns:
            dict: The JSON response from the Syzygy tablebase server if successful.
        
        Raises:
            requests.RequestException: If the request to the Syzygy tablebase server fails.
        """

        fen = board.fen()
        fen_encoded = urllib.parse.quote(fen)

        url = self.SYGYZY_URL + fen_encoded

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.RequestException(f"Request to Syzygy tablebase server failed with status code {response.status_code}")

    def best_sygyzy(self, board: chess.Board) -> chess.Move | None:
        """
        Get the best move from the Syzygy tablebase server for the given board position.

        Args:
            board (chess.Board): The chess board position to get the best move for.

        Returns:
            chess.Move: The best move from the Syzygy tablebase server.
            None: If no best move is found.
        """

        num_pieces = len(board.piece_map().values())
        if num_pieces > 7: # Syzygy only supports up to 7 pieces
            return None

        response = self.sygyzy_query(board)
        print([move["uci"] for move in response["moves"]])
        return chess.Move.from_uci(response["moves"][0]["uci"])

    def opening_query(self, board: chess.Board) -> dict:
        """
        Query the opening book for the best move in the given board position.

        Args:
            board (chess.Board): The chess board position to query.

        Returns:
            dict: The JSON response from the opening book server if successful.

        Raises:
            requests.RequestException: If the request to the opening book server fails.
        """

        fen = board.fen()
        fen_encoded = urllib.parse.quote(fen)

        url = self.OPENING_URL + fen_encoded

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.RequestException(f"Request to opening book server failed with status code {response.status_code}")

    def random_opening_move(self, board: chess.Board) -> chess.Move | None:
        """
        Get a random move from the opening book for the given board position.

        Args:
            board (chess.Board): The chess board position to get a random opening move for.

        Returns:
            chess.Move: A random move from the opening book.
            None: If no moves are available in the opening book or the opening leave chance is not met.
        """

        odds = 1 - (1 - self.OPENING_LEAVE_CHANCE) ** board.fullmove_number
        if rnd.random() < odds and board.fullmove_number > 7:
            return None

        try:
            response = self.opening_query(board)
        except requests.RequestException as e:
            print(f"Error querying opening book: {e}")
            return None
        
        if "moves" in response and response["moves"]:
            moves = response["moves"]

            weights = [move["white"] + move["black"] + move["draws"] for move in moves]
            chosen_move = rnd.choices(moves, weights=weights, k=1)[0]
            return chess.Move.from_uci(chosen_move["uci"])
        return None

    ##################################################
    #                   HEURISTICS                   #
    ##################################################

    def _score_weak_heuristic(self, board: chess.Board, weak_depth: int = 0) -> list[tuple[chess.Move, float]]:
        """
        Evaluate and score each legal move from the current board position using the Minimax algorithm 
        without heuristic sorting or elimination.

        :param board: The current state of the chess board.
        :type board: chess.Board
        :param weak_depth: The depth to search in the Minimax algorithm.
        :type weak_depth: int
        :return: A list of tuples containing legal moves and their corresponding scores.
        :rtype: list[tuple[chess.Move, float]]
        """

        move_score_map: list[tuple[chess.Move, float]] = []

        for move in board.legal_moves:
            board.push(move)
            score = self.minimax(board, weak_depth, float('-inf'), float('inf'), heuristic_sort=False, heuristic_eliminate=False)
            board.pop()
            move_score_map.append((move, score))
        
        return move_score_map

    def weak_heuristic_moves(self, board: chess.Board, depth: int = 0) -> list[chess.Move]:
        """
        Generate a list of moves in weak heuristic order.

        :param board: The current state of the board
        :type board: chess.Board
        :param depth: The depth to search in the Minimax algorithm
        :type depth: int
        :return: A list of moves in weak heuristic order
        :rtype: list[chess.Move]
        """

        move_score_map = self._score_weak_heuristic(board, depth)
        
        # Sort moves by score
        sorted_moves = sorted(move_score_map, key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        return [move for move, _ in sorted_moves]
    
    def _turning_point(self, scores: list[float]) -> int:
        """
        Find the index of the 'turning point' in sorted scores by identifying the biggest gap.

        The method takes a sorted list of scores as input and returns the index after the biggest gap.
        If the biggest gap is smaller than a threshold fraction of the score range, it returns len(scores) to keep all moves.

        :param scores: A sorted list of scores
        :return: The index of the turning point or len(scores) if no big gap found
        """

        if not scores:
            return -1
        if len(scores) == 1:
            return 0

        score_range = max(scores) - min(scores)
        if score_range == 0:
            return len(scores)

        threshold_fraction = 0.15  # 15% of the score range

        max_gap = -1
        max_gap_index = 0

        for i in range(len(scores) - 1):
            gap = abs(scores[i] - scores[i + 1])
            if gap > max_gap:
                max_gap = gap
                max_gap_index = i

        if max_gap / score_range < threshold_fraction:
            return len(scores)

        return max_gap_index + 1

    def select_wh_moves(self, board: chess.Board, depth: int = 0) -> list[chess.Move]:
        """
        Select some moves worth exploring based on a weak heuristic.

        :param board: The current state of the board
        :type board: chess.Board
        :param depth: The depth to search in the Minimax algorithm
        :type depth: int
        :return: A list of moves in weak heuristic order
        :rtype: list[chess.Move]
        """

        move_score_map = self._score_weak_heuristic(board, depth)
        
        # Sort moves by score
        sorted_moves = sorted(move_score_map, key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        
        # Decide turning point (elbow function)
        scores = [score for _, score in sorted_moves]
        cutoff_index = self._turning_point(scores)
        selected_moves = [move for move, _ in sorted_moves[:cutoff_index+1]]

        return selected_moves

    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """
        Calculate the MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score for a move.

        :param board: The current state of the board
        :param move: The move to score
        :return: An integer score for move ordering
        """
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None or attacker is None:
            return 0

        victim_value = self.MATERIAL.get(victim.piece_type, 0)
        attacker_value = self.MATERIAL.get(attacker.piece_type, 0)

        # Higher score for capturing more valuable victim with less valuable attacker
        return (victim_value * 10) - attacker_value

    def mvv_lva_ordering(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Order moves using MVV-LVA heuristic.

        :param board: The current state of the board
        :param moves: List of moves to order
        :return: List of moves ordered by MVV-LVA score descending
        """
        scored_moves = [(move, self.mvv_lva_score(board, move)) for move in moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in scored_moves]

    ##################################################
    #                   EVALUATION                   #
    ##################################################

    HEATMAP_PATH = "chess/global-assets/heatmap.json"
    HEATMAP = json.load(open(HEATMAP_PATH))

    MATERIAL: dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 25
    }

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, *, original_depth: int = 0, heuristic_sort: bool = True, heuristic_eliminate: bool = True, use_mvv_lva: bool = False) -> float:
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
        :param original_depth: The original depth of the search
        :type original_depth: int
        :param heuristic_sort: Whether to sort moves by heuristic score
        :type heuristic_sort: bool
        :param heuristic_eliminate: Whether to eliminate moves with low heuristic scores
        :type heuristic_eliminate: bool
        :param use_mvv_lva: Whether to order moves using MVV-LVA heuristic
        :type use_mvv_lva: bool
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

        # Check for claimable draw conditions and treat as terminal draw state
        if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
            return 0.0

        if depth == 0 or board.is_game_over() or self.is_timeup():
            return self.evaluate(board)


        is_maximizing = board.turn == chess.WHITE
        best_score = float('-inf') if is_maximizing else float('inf')
        best_move = None

        search_depth = original_depth - depth

        # Null Move Pruning (NMP)
        R = 2  # Reduction for null move pruning
        if depth > 2 and not board.is_check():
            board.push(chess.Move.null())
            null_score = -self.minimax(board, depth - 1 - R, -beta, -beta + 1, original_depth=original_depth, heuristic_sort=heuristic_sort, heuristic_eliminate=heuristic_eliminate, use_mvv_lva=use_mvv_lva)
            board.pop()
            if null_score >= beta:
                return null_score

        # Move ordering with Killer Move Heuristics (KMH) and MVV-LVA prioritization
        legals = list(board.legal_moves)

        # Prioritize killer moves at this depth
        killer_moves_at_depth = self.killer_moves.get(search_depth, [])

        # Separate killer moves and other moves
        killer_moves_in_legals = [move for move in killer_moves_at_depth if move in legals]
        other_moves = [move for move in legals if move not in killer_moves_in_legals]

        # Order other moves with MVV-LVA if enabled
        if use_mvv_lva:
            other_moves = self.mvv_lva_ordering(board, other_moves)

        # Combine killer moves first, then other moves
        ordered_moves = killer_moves_in_legals + other_moves

        for move in ordered_moves:
            board.push(move)
            score = self.minimax(board, depth - 1, alpha, beta, heuristic_sort=heuristic_sort, original_depth=original_depth, heuristic_eliminate=heuristic_eliminate, use_mvv_lva=use_mvv_lva)
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

            # Update killer moves on beta cutoff with non-capturing moves
            if beta <= alpha:
                if not board.is_capture(move):
                    # Add move to killer moves for this depth if not already present
                    if search_depth not in self.killer_moves:
                        self.killer_moves[search_depth] = []
                    if move not in self.killer_moves[search_depth]:
                        self.killer_moves[search_depth].append(move)
                        # Limit to 2 killer moves per depth
                        if len(self.killer_moves[search_depth]) > 2:
                            self.killer_moves[search_depth].pop(0)
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

        piece_map = board.piece_map()

        # Try to get score from DB
        cached_score = self.evaluate_from_db(board)
        if cached_score is not None:
            return cached_score

        # Game over
        if board.is_game_over():
            # Explicitly check for draw conditions
            if board.is_checkmate():
                if board.result() == "1-0":
                    return float('inf')
                elif board.result() == "0-1":
                    return float('-inf')
            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
                return 0
            # Fallback for other game over conditions
            return 0
        
        stage = self.get_game_stage(board)

        PAWNS = {chess.WHITE: [], chess.BLACK: []}
        for square, piece in piece_map.items():
            if piece.piece_type == chess.PAWN:
                PAWNS[piece.color].append(square)

        KINGS = {chess.WHITE : board.king(chess.WHITE), chess.BLACK : board.king(chess.BLACK)}

        def evaluate_player(color: chess.Color) -> float:

            king_square = KINGS[color]
            enemy_king_square = KINGS[not color]
            if king_square is None or enemy_king_square is None:
                return 0

            def coverage() -> float:
                attack_bonus = 0
                cover_bonus = 0

                # Reward squares covered
                attacked_squares: list[chess.Square] = []
                for square, piece in piece_map.items():
                    if piece.color == color:
                        attacks = board.attacks(square)
                        attacked_squares.extend(list(attacks))

                        # Reward coverage (reduce score for coverage by high value pieces since their control is weaker)
                        cover_bonus += len(attacked_squares) / self.MATERIAL[piece.piece_type]
                
                for square in set(attacked_squares):
                    # Exponentially reward squares attacked by multiple pieces
                    attack_bonus += attacked_squares.count(square) ** 1.25

                    piece = board.piece_at(square)
                    rank = chess.square_rank(square)

                    # Reward centre control
                    if square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                        attack_bonus += 2.5 * aggression[color]
                    # Reward control of enemy half
                    if color == chess.WHITE and rank > 3:
                        attack_bonus += 1.5 * aggression[color]
                    elif color == chess.BLACK and rank < 4:
                        attack_bonus += 1.5 * aggression[color]

                    if piece is None:
                        continue

                    # Reward pieces in the enemy half
                    if color == chess.WHITE and rank > 3:
                        attack_bonus += self.MATERIAL[piece.piece_type] * 2 ** 0.6 * aggression[color]
                    elif color == chess.BLACK and rank < 4:
                        attack_bonus += self.MATERIAL[piece.piece_type] * 2 ** 0.6 * aggression[color]

                    # Give bonuses for attacking high value pieces, give bonuses to defending low value pieces
                    if piece.color != color:
                        attack_bonus += self.MATERIAL[piece.piece_type] ** 1.5 * aggression[color]
                    else:
                        attack_bonus += self.MATERIAL[piece.piece_type] * 2 ** 0.9 * aggression[not color]
                
                return attack_bonus ** 0.5 + cover_bonus
            
            def control() -> float:
                # Reward pieces per square control
                control_bonus = 0
                for square in piece_map.keys():
                    attacks = board.attacks(square)
                    control_bonus += len(attacks) ** 0.35 # type: ignore

                    if square not in [chess.E4, chess.E5, chess.D4, chess.D5]:
                        continue
                    piece = board.piece_at(square)
                    if piece is None or piece.color != color:
                        continue
                    control_bonus += self.MATERIAL[piece.piece_type]
                    if piece == chess.PAWN:
                        control_bonus += 4
                    elif piece == chess.KNIGHT:
                        control_bonus += 2

                return control_bonus
        
            def material_score() -> float:
                # Base material score
                base_score = material[color]

                # Add stronger dependency based on how many squares each piece takes up (mobility)
                mobility_score = 0
                for square, piece in piece_map.items():
                    if piece.color == color:
                        attacks = board.attacks(square)
                        mobility_count = len(attacks)
                        # Weight mobility more heavily, with nonlinear scaling
                        piece_weight = {
                            chess.PAWN: 0.5,
                            chess.KNIGHT: 1.5,
                            chess.BISHOP: 1.7,
                            chess.ROOK: 1.2,
                            chess.QUEEN: 1.0,
                            chess.KING: 0.3
                        }.get(piece.piece_type, 1.0)
                        mobility_score += (mobility_count ** 0.75) * piece_weight * 0.5

                return base_score + mobility_score

            def heatmap() -> float:
                heatmap_score: float = 0
                for square, piece in piece_map.items():
                    if piece.color != color:
                        continue
                    piece_symbol = piece.symbol().upper()

                    # Convert square index to rank and file
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)

                    # Heatmaps are incorrectly oriented for python chess; flip them
                    if piece.color == chess.WHITE:
                        rank = 7 - rank
                    heatmap_score += self.HEATMAP[stage][piece_symbol][rank][file]
                
                return heatmap_score
            
            def low_legal_penalty() -> float:
                # Punish low number of legal moves
                if color == board.turn:
                    legal_moves = board.legal_moves
                else:
                    board.turn = color
                    legal_moves = board.legal_moves
                    board.turn = not color
                
                return 1 / len(list(legal_moves))

            def minor_piece_bonus() -> float:
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
                
                return minor_piece_development_bonus

            def king_safety_penalty() -> float:
                # King safety penalties
                king_penalty = 0
                # Since python-chess does not have has_castled method, check castling rights and castling status differently
                # We can check if the castling rights are lost but the king is still on the original square, meaning no castling yet
                king_start_square = chess.E1 if color == chess.WHITE else chess.E8
                has_moved = king_square != king_start_square
                if stage != 'late' and (not board.has_castling_rights(color)) and has_moved:
                    king_penalty += 10.0
                king_moves = list(board.attacks(king_square))
                king_penalty -= len(king_moves) ** 0.5 # Penalise less for more king moves

                return king_penalty

            def pawn_structure() -> float:

                pawn_score = 0
                pawns = PAWNS[color]
                enemy_pawns = PAWNS[not color]

                pawn_files = [chess.square_file(sq) for sq in pawns]
                enemy_pawn_files = [chess.square_file(sq) for sq in enemy_pawns]

                for square in pawns:
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)

                    # Penalize doubled pawns: more than one pawn on the same file
                    if pawn_files.count(file) > 1:
                        pawn_score -= 3.0  # Increased penalty

                    # Penalize isolated pawns: no friendly pawns on adjacent files
                    if (file - 1 not in pawn_files) and (file + 1 not in pawn_files):
                        pawn_score -= 3.0  # Increased penalty

                    # Penalize backward pawns: pawns that cannot be defended by other pawns and are behind the pawn chain
                    is_backward = False
                    if color == chess.WHITE:
                        for adj_file in [file - 1, file + 1]:
                            if 0 <= adj_file <= 7:
                                adj_squares = [chess.square(adj_file, r) for r in range(rank)]
                                if not any(sq in pawns for sq in adj_squares):
                                    is_backward = True
                    else:
                        for adj_file in [file - 1, file + 1]:
                            if 0 <= adj_file <= 7:
                                adj_squares = [chess.square(adj_file, r) for r in range(rank + 1, 8)]
                                if not any(sq in pawns for sq in adj_squares):
                                    is_backward = True
                    if is_backward:
                        pawn_score -= 2.0

                    # Penalize hanging pawns: pawns that are undefended and can be captured easily
                    piece = board.piece_at(square)
                    if piece is not None:
                        defenders = board.attackers(color, square)
                        if len(defenders) == 0:
                            pawn_score -= 2.5

                    # Reward connected pawns: pawns on adjacent files and ranks
                    connected = False
                    for adj_file in [file - 1, file + 1]:
                        for adj_rank in [rank - 1, rank, rank + 1]:
                            if 0 <= adj_file <= 7 and 0 <= adj_rank <= 7:
                                adj_square = chess.square(adj_file, adj_rank)
                                piece = board.piece_at(adj_square)
                                if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                                    connected = True
                                    break
                        if connected:
                            break
                    if connected:
                        pawn_score += 1.0

                    # Reward passed pawns: no enemy pawns in front or adjacent files ahead
                    is_passed = True
                    enemy_pawn_ranks = [chess.square_rank(sq) for sq in enemy_pawns]
                    for ep_file, ep_rank in zip(enemy_pawn_files, enemy_pawn_ranks):
                        if abs(ep_file - file) <= 1:
                            if (color == chess.WHITE and ep_rank > rank) or (color == chess.BLACK and ep_rank < rank):
                                is_passed = False
                                break
                    if is_passed:
                        pawn_score += 2.0

                    # Reward pawns close and in front of allied king: king distance = 1
                    if chess.square_distance(square, king_square) <= 1:
                        pawn_score += 1.0

                return pawn_score

            def attack_quality() -> float:

                aggression_score = 0
                # Reward high-material pieces for being close to the enemy king
                # Get distance from enemy king for each piece
                for square, piece in piece_map.items():
                    if piece.color == color:
                        aggression_score += self.MATERIAL[piece.piece_type] / (chess.square_distance(square, enemy_king_square)) * 5

                # Reward / penalise checks
                if board.is_check():
                    if board.turn == color:
                        aggression_score -= 1
                    else:
                        aggression_score += 1

                return aggression_score

            score = 0            
            score -= king_safety_penalty() * 4
            score -= (4 * low_legal_penalty()) ** 1.5 * (aggression[not color] ** 2)
            score += (material_score() ** 2 * 20)
            score += (coverage() ** 1.1 * 0.1) * aggression[color]
            score += heatmap() ** (3 if stage == 'early' else 1) * aggression[not color] * 3 * (10 if stage == 'late' else 7.5 if stage == 'early' else 5)
            score += (control() * 1.25) * 0.35 * aggression[color] * (2 if stage == 'late' else 1.5 if stage == 'early' else 1)
            score += minor_piece_bonus() * 15 * aggression[color]
            score += pawn_structure() * 15 * (2 if stage == 'late' else 1.5 if stage == 'early' else 1)
            score += attack_quality() ** 1.2 * aggression[color] * 15

            # print("\nAGG", aggression[color])
            # print("EAG", aggression[not color])
            # print("-KSP", king_safety_penalty() * 2)
            # print("-QPS", queen_penalty_score() / aggression[color])
            # print("-LLP", low_legal_penalty() ** 1.5 * aggression[not color] ** 2)
            # print("MS", material_score() ** 2 * 10)
            # print("ATK", attack() ** 1.5 * 0.1 * aggression[color])
            # print("HT", heatmap() ** 3 * aggression[not color])
            # print("CTL", control() ** 1.5 * 0.35 * aggression[color])
            # print("MPC", minor_piece_bonus() * 10 * aggression[color])
            # print("PS", pawn_structure() * 10)
            # print("AQ", attack_quality() * aggression[color] * 20)
            # print("FINAL",score)
            # input()

            return score
        
        # Material
        material = {chess.WHITE: 1, chess.BLACK: 1} # 1 to avoid division by 0
        for _, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue
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

        self.conn = sqlite3.connect(self.TRANSPOSITION_PATH)
        self.cursor = self.conn.cursor()

        print(f"{"white" if board.turn == chess.WHITE else "black"} move")

        self.start_time = time.time()
        self.timeout = timeout

        # First try a random opening move
        opening_best = self.random_opening_move(board)
        if opening_best is not None:
            print("Using random opening move")
            self.conn.close()
            return opening_best

        # Get Sygyzy best move
        syg_best = self.best_sygyzy(board)
        if syg_best is not None:
            print("Using Sygzy best move")
            self.conn.close()
            return syg_best


        # Check if there is a stored winning move for the current position
        stored_move = get_stored_winning_move(board)
        if stored_move is not None and stored_move in board.legal_moves:
            print("Using stored winning move")
            self.conn.close()
            return stored_move

        depth = 1
        best_move = None
        save = True

        maxmin = max if board.turn == chess.WHITE else min

        moves = list(board.legal_moves)  # Convert generator to list for membership checks

        move_score_map: list[tuple[chess.Move, float]] = [(move, 0) for move in moves] # Initialize once before loop

        while not self.is_timeup():

            print(f"""DEPTH {depth}: """,end='\t')

            move_score_map.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
            moves = [move for move, _ in move_score_map]

            # Gradually filter out based on the previous scores
            if depth > 2:
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

            current_best_move = None
            current_best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

            # Evaluate each mmove
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
                    # Check if the move has already been evaluated
                    score = self.evaluate_from_db(board, depth)

                    if score is not None:
                        board.pop()
                        move_score_dict[move] = score
                        # Update current best if needed
                        if board.turn == chess.WHITE:
                            if score > current_best_score:
                                current_best_score = score
                                current_best_move = move
                        else:
                            if score < current_best_score:
                                current_best_score = score
                                current_best_move = move
                        continue
                
                # Minimax
                score = self.minimax(board, depth, float('-inf'), float('inf'), original_depth=depth, heuristic_eliminate=False, use_mvv_lva=True)
                board.pop()

                if self.is_timeup():
                    print("TIMEUP")
                    save = False
                    break

                print(f"{board.san(move)} : {score:.2f}",end='\t',flush=True)
                
                move_score_dict[move] = score

                # Update current best if needed
                if board.turn == chess.WHITE:
                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move
                else:
                    if score < current_best_score:
                        current_best_score = score
                        current_best_move = move

                self.save_evaluation(board, score, depth)

                if _should_terminate(list(move_score_dict.items())):
                    print("TERMINATED EARLY")
                    break
            
            print()

            # Update move_score_map from the dictionary for next iteration
            move_score_map = list(move_score_dict.items())

            # Terminate early if an immediate win is found
            if _should_terminate(move_score_map) or all(score == self.WORST_SCORE for _, score in move_score_map):
                # Choose best move from current depth
                best_move = current_best_move if current_best_move is not None else maxmin(move_score_map, key=lambda x: x[1])[0]

                # Save the winning move and the position before the move
                save_winning_move(board, best_move)
                break

            best_move = current_best_move if current_best_move is not None else maxmin(move_score_map, key=lambda x: x[1])[0]

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

        num_pieces = len([piece for piece in board.piece_map().values() if piece.piece_type != chess.PAWN])
        if num_pieces >= 12:
            return "early"
        elif num_pieces >= 8:
            return "middle"
        else:
            return "late"

    def is_timeup(self) -> bool:
        if self.timeout is None or self.start_time is None:
            return False
        return time.time() - self.start_time > self.timeout

def main():

    # FEN = "8/1p6/ppp3kP/6P1/1K3P2/4P3/8/8 w - - 0 1"
    FEN = chess.STARTING_FEN

    board = chess.Board(FEN)
    players = [Computer(board.turn), Computer(not board.turn)]

    while not board.is_game_over():
        print(board,"\n\n")
        player = players[0] if board.turn == chess.WHITE else players[1]
        move = player.best_move(board, timeout=20)
        if move is None:
            break
        print(f"\n\n{board.fullmove_number}. {board.san(move)}")
        board.push(move)
    print(board)
    print("GAME OVER!")

if __name__ == "__main__":
    main()
