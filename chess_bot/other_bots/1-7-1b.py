import chess
import chess.polyglot
import os
import random as rnd
import requests
import sqlite3
import time
import urllib.parse

__version__ = '1.7.1b'

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

        # Metrics
        self.nodes_explored = 0
        self.leaf_nodes_explored = 0
        self.alpha_cuts = 0
        self.beta_cuts = 0
        self.prunes = 0
        self.cache_hits = 0

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
        
        # Create directory if it doesn't exist
        if not os.path.exists("chess/tables"):
            os.makedirs("chess/tables")

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
                position_zobrist TEXT PRIMARY KEY,
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
        
        zobrist_key = str(chess.polyglot.zobrist_hash(board))
        self.cursor.execute("SELECT score FROM transposition_table WHERE zobrist_key = ? AND depth >= ?", (zobrist_key, depth,))
        row = self.cursor.fetchone()
        if row is not None:
            self.cache_hits += 1
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
        
        zobrist_key = str(chess.polyglot.zobrist_hash(board))
        try:
            # Use UPSERT to handle both insert and update in a single query
            self.cursor.execute("""
                INSERT INTO transposition_table (zobrist_key, score, depth) 
                VALUES (?, ?, ?)
                ON CONFLICT(zobrist_key) 
                DO UPDATE SET 
                    score = CASE 
                        WHEN excluded.depth > transposition_table.depth THEN excluded.score 
                        ELSE transposition_table.score 
                    END,
                    depth = CASE 
                        WHEN excluded.depth > transposition_table.depth THEN excluded.depth 
                        ELSE transposition_table.depth 
                    END
            """, (zobrist_key, score, depth))
        except sqlite3.Error as e:
            print(f"Error saving evaluation to DB: {e}")

    def get_stored_winning_move(self, board: chess.Board) -> chess.Move | None:
        zobrist_key = str(chess.polyglot.zobrist_hash(board))
        self.cursor.execute("SELECT move_uci FROM winning_moves WHERE position_zobrist = ?", (zobrist_key,))
        row = self.cursor.fetchone()
        if row is not None:
            try:
                return chess.Move.from_uci(row[0])
            except:
                return None
        return None

    def save_winning_move(self, board_before_move: chess.Board, move: chess.Move) -> None:
        zobrist_key = str(chess.polyglot.zobrist_hash(board_before_move))
        move_uci = move.uci()
        try:
            self.cursor.execute("INSERT OR REPLACE INTO winning_moves (position_zobrist, move_uci) VALUES (?, ?)", (zobrist_key, move_uci))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving winning move to DB: {e}")


    ##################################################
    #            OPENING BOOKS AND SYGYZY            #
    ##################################################

    SYGYZY_URL = "https://tablebase.lichess.ovh/standard?fen="
    OPENING_URL = "https://explorer.lichess.ovh/master?fen="

    OPENING_LEAVE_CHANCE = 0.05  # Chance to leave the opening book

    def can_sygyzy(self, board: chess.Board, best_score: float) -> bool:
        num_pieces = bin(board.occupied).count('1')
        if num_pieces > 7:
            return False

        win_threshold = float('inf') * (1 if board.turn == chess.WHITE else -1)
        lose_threshold = 1e5 * (-1 if board.turn == chess.WHITE else 1)
        if best_score >= win_threshold or best_score <= lose_threshold:
            return False

        return True

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

    def best_sygyzy(self, board: chess.Board, best_score: float) -> chess.Move | None:
        """
        Get the best move from the Syzygy tablebase server for the given board position.

        Args:
            board (chess.Board): The chess board position to get the best move for.
            best_score (float): The best score for the board position.

        Returns:
            chess.Move: The best move from the Syzygy tablebase server.
            None: If no best move is found.
        """

        if not self.can_sygyzy(board, best_score):
            return None

        response = self.sygyzy_query(board)
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

        if self.get_game_stage(board.piece_map()) == "late":
            return {"moves": []}

        fen = board.fen()
        fen_encoded = urllib.parse.quote(fen)

        url = self.OPENING_URL + fen_encoded

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(3)
                return self.opening_query(board)
            else:
                raise requests.RequestException(f"Request to opening book server failed with status code {response.status_code}")

        except requests.ConnectionError: # If max retries, no response etc, ignore
            return {"moves": []}

    def random_opening_move(self, board: chess.Board) -> chess.Move | None:
        """
        Get a random move from the opening book for the given board position.

        Args:
            board (chess.Board): The chess board position to get a random opening move for.

        Returns:
            chess.Move: A random move from the opening book.
            None: If no moves are available in the opening book or the opening leave chance is not met.
        """

        odds = 1 - (1 - self.OPENING_LEAVE_CHANCE) ** (board.fullmove_number / 2)
        if rnd.random() < odds and board.fullmove_number > 5:
            return None

        response = self.opening_query(board)
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
    
    def _turning_point(self, scores: list[float], threshold: float=0.25) -> int:
        """
        Find the index of the 'turning point' in sorted scores by identifying the first gap that meets the threshold.

        The method takes a sorted list of scores as input and returns the index after the first gap that is greater than
        a threshold fraction of the score range.

        :param scores: A sorted list of scores
        :param threshold: The threshold fraction of the score range
        :return: The index of the turning point or len(scores) if no qualifying gap is found
        """

        if not scores:
            return -1
        if len(scores) == 1:
            return 0

        score_range = max(scores) - min(scores)
        if score_range == 0:
            return len(scores) // 2

        for i in range(len(scores) - 1):
            gap = abs(scores[i] - scores[i + 1])
            # Check if the gap is greater than the threshold and greater than 50 centipawns
            if gap / score_range >= threshold and gap > self.ESTIMATED_PAWN_VALUE * 0.50:
                return i + 1

        return len(scores)

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
        victim = board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)

        if victim is None or attacker is None:
            return 0

        victim_value = self.MATERIAL.get(victim, 0)
        attacker_value = self.MATERIAL.get(attacker, 0)

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

    HEATMAP = {
        "early": {
            "P": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0], [1.0, 1.5, 2.0, 3.5, 3.5, 2.0, 1.5, 1.0], [0.0, -0.5, -0.25, 3.0, 3.0, -0.25, -0.5, 0.0], [0.0, -0.5, -1.0, -0.5, -0.5, -1.0, -0.5, 0.0], [1.0, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "N": [[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0], [-4.0, -2.0, -0.5, -0.5, -0.5, -0.5, -2.0, -4.0], [-3.0, 0.0, 0.5, 0.75, 0.75, 0.5, 0.0, -3.0], [-3.0, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -3.0], [-3.0, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -3.0], [-3.0, 0.0, 0.5, 0.75, 0.75, 0.5, 0.0, -3.0], [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0], [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]],
            "B": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [-1.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, -1.0], [-1.0, 0.125, 0.5, 0.5, 0.5, 0.5, 0.125, -1.0], [-1.0, 0.25, 0.875, 0.5, 0.5, 0.875, 0.25, -1.0], [-1.0, 0.375, 0.625, 0.75, 0.75, 0.625, 0.375, -1.0], [-1.5, 0.5, 0.125, 0.25, 0.25, 0.125, 0.5, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
            "R": [[-0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.25], [0.5, 1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.75, 0.25, 0.375, 0.75, 0.75, 0.375, 0.25, -0.5], [-1.0, -0.75, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0]],
            "Q": [[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]],
            "K": [[-4, -5, -5, -5, -5, -5, -5, -4], [-4, -5, -5, -5, -5, -5, -5, -4], [-3, -5, -5, -5, -5, -5, -5, -3], [-3, -4, -5, -5, -5, -5, -4, -3], [-2, -3, -4, -5, -5, -4, -3, -2], [-1, -2, -3, -4, -4, -3, -2, -1], [0, -1, -2, -3, -3, -2, -1, 0], [2, 3, 2, 0, 0, 2, 4, 2]]
        },
        "middle": {
            "P": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0], [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0], [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0], [0.5, 0.0, -0.5, 2.0, 2.0, -0.5, 0.0, 0.5], [0.5, 0.25, -1.0, 1.0, 1.0, -1.0, 0.25, 0.5], [1.0, 1.0, 1.0, -1.5, -1.5, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "N": [[-4.0, -3.0, -2.0, -2.0, -2.0, -2.0, -3.0, -4.0], [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0], [-2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, -2.0], [-2.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, -2.0], [-2.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, -2.0], [-2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, -2.0], [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0], [-4.0, -3.0, -2.0, -2.0, -2.0, -2.0, -3.0, -4.0]],
            "B": [[-1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -1.0, -1.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -0.5], [-0.5, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -0.5], [-0.5, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -0.5], [-0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -0.5], [-1.0, 0.625, 0.5, 0.5, 0.5, 0.5, 0.625, -1.0], [-1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -1.0, -1.5]],
            "R": [[-0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.25], [0.5, 1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.75, 0.25, 0.375, 0.75, 0.75, 0.375, 0.25, -0.5], [-1.0, -0.75, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0]],
            "Q": [[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]],
            "K": [[-10.0, -9.0, -8.0, -8.0, -8.0, -8.0, -9.0, -10.0], [-7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0], [-5.0, -5.0, -6.0, -6.0, -6.0, -6.0, -5.0, -5.0], [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0], [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0], [0.5, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.5], [1.0, 2.0, 3.0, -1.0, -1.0, 0.0, 3.0, 1.0]]
        },
        "late": {
            "P": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.5, -1.5, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "N": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, -0.875, 0.0, 1.0, 1.0, 0.0, -0.875, -1.5], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, -1.0], [-1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.5, -0.875, 0.0, 0.5, 0.5, 0.0, -0.875, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
            "B": [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]],
            "R": [[-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [1.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.25, 1.25, 1.0, 0.0, -0.5], [-0.5, 0.0, 0.75, 1.0, 1.0, 0.75, 0.0, -0.5], [-1.0, -0.5, 0.5, 0.75, 0.75, 0.5, -0.5, -1.0]],
            "Q": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
            "K": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1.5], [-1.0, -0.5, 1.5, 1.5, 1.5, 1.5, -0.5, -1.0], [-1.0, -0.5, 1.5, 2.5, 2.5, 1.5, -0.5, -1.0], [-1.0, -0.5, 1.5, 2.5, 2.5, 1.5, -0.5, -1.0], [-1.0, -0.5, 1.5, 1.5, 1.5, 1.5, -0.5, -1.0], [-1.5, -1.25, -1.0, -0.75, -0.75, -1.0, -1.25, -1.5], [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]]
        }
    }

    MATERIAL: dict[int, int] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 25
    }

    ESTIMATED_PAWN_VALUE = 20000

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
        :return: The best score possible for the maximizing player, or None if timeout occurred
        :rtype: float
        """

        self.nodes_explored += 1

        # Check for timeout at the start of evaluation
        if self.is_timeup():
            return float('nan')  # Return NaN to indicate timeout

        # Instant game over checks
        if board.is_checkmate():
            # If checkmate, winner is the opponent of the current turn
            if board.turn == chess.WHITE:
                return float('-inf')
            else:
                return float('inf')
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_repetition(count=3):
            return 0

        if depth == 0:
            self.leaf_nodes_explored += 1
            return self.evaluate(board)

        is_maximizing = board.turn == chess.WHITE
        best_score = float('-inf') if is_maximizing else float('inf')
        best_move = None

        # Futility pruning
        # Only apply futility pruning when not in check and at frontier nodes
        if depth <= 2 and not board.is_check():
            # Use a margin based on ESTIMATED_PAWN_VALUE
            margin = self.ESTIMATED_PAWN_VALUE * (5 if depth == 2 else 3)
            
            # Get static evaluation of current position
            static_eval = self.evaluate(board)
            
            # For maximizing player (white)
            if is_maximizing and static_eval + margin < alpha:
                self.prunes += 1
                self.alpha_cuts += 1
                return static_eval + margin
            
            # For minimizing player (black)
            elif not is_maximizing and static_eval - margin > beta:
                self.prunes += 1
                self.beta_cuts += 1
                return static_eval - margin

        search_depth = original_depth - depth

        # Null Move Pruning (NMP)
        R = 2  # Reduction for null move pruning
        if depth > 2 and not board.is_check():
            board.push(chess.Move.null())
            null_score = -self.minimax(board, depth - 1 - R, -beta, -beta + 1, original_depth=original_depth, heuristic_sort=heuristic_sort, heuristic_eliminate=heuristic_eliminate, use_mvv_lva=use_mvv_lva)
            board.pop()
            if null_score >= beta:
                self.beta_cuts += 1
                self.prunes += 1
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

                self.prunes += len(ordered_moves) - ordered_moves.index(move)
                self.alpha_cuts += 1
                break

        # If this position leads to a winning score, save the winning move
        # Check if the move leads to checkmate for the opponent
        # When best_score is +inf, it means the maximizing player (White) wins
        # When best_score is -inf, it means the minimizing player (Black) wins
        winning_for_bot = False
        
        if self.color == chess.WHITE and best_score == float('inf'):
            # Bot is White and this move leads to White checkmating Black
            winning_for_bot = True
        elif self.color == chess.BLACK and best_score == float('-inf'):
            # Bot is Black and this move leads to Black checkmating White
            winning_for_bot = True
            
        if winning_for_bot and best_move is not None:
            self.save_winning_move(board, best_move)

        return best_score

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate the board state and return a score.

        :param board: The current state of the board
        :type board: chess.Board
        :return: A numerical evaluation of the board state
        :rtype: float
        """

        def cse(x: float, y: float) -> float:
            """Complex safe exponentiation."""
            if x > 0:
                return x ** y
            else:
                return -(abs(x) ** y)

        # Try to get score from DB cache
        cached_score = self.evaluate_from_db(board)
        if cached_score is not None:
            return cached_score

        # Use bitboards for efficient piece counting and material calculation
        white_pieces = board.occupied_co[chess.WHITE]
        black_pieces = board.occupied_co[chess.BLACK]
        
        # Calculate material using bitboard population counts
        material = {chess.WHITE: 1, chess.BLACK: 1}  # Avoid division by zero
        
        # Count pieces by type using bitboards
        white_pawns = board.pawns & white_pieces
        white_knights = board.knights & white_pieces
        white_bishops = board.bishops & white_pieces
        white_rooks = board.rooks & white_pieces
        white_queens = board.queens & white_pieces
        
        black_pawns = board.pawns & black_pieces
        black_knights = board.knights & black_pieces
        black_bishops = board.bishops & black_pieces
        black_rooks = board.rooks & black_pieces
        black_queens = board.queens & black_pieces
        
        # Calculate material scores using bitboard population counts
        material[chess.WHITE] += bin(white_pawns).count('1') * self.MATERIAL[chess.PAWN]
        material[chess.WHITE] += bin(white_knights).count('1') * self.MATERIAL[chess.KNIGHT]
        material[chess.WHITE] += bin(white_bishops).count('1') * self.MATERIAL[chess.BISHOP]
        material[chess.WHITE] += bin(white_rooks).count('1') * self.MATERIAL[chess.ROOK]
        material[chess.WHITE] += bin(white_queens).count('1') * self.MATERIAL[chess.QUEEN]
        
        material[chess.BLACK] += bin(black_pawns).count('1') * self.MATERIAL[chess.PAWN]
        material[chess.BLACK] += bin(black_knights).count('1') * self.MATERIAL[chess.KNIGHT]
        material[chess.BLACK] += bin(black_bishops).count('1') * self.MATERIAL[chess.BISHOP]
        material[chess.BLACK] += bin(black_rooks).count('1') * self.MATERIAL[chess.ROOK]
        material[chess.BLACK] += bin(black_queens).count('1') * self.MATERIAL[chess.QUEEN]
        
        # Build PIECES dict for compatibility (only when needed)
        piece_map = board.piece_map()
        PIECES = {chess.WHITE: {}, chess.BLACK: {}}
        for square, piece in piece_map.items():
            PIECES[piece.color][square] = piece

        stage = self.get_game_stage(piece_map)

        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            return 0
        KINGS = {chess.WHITE: white_king, chess.BLACK: black_king}

        # Precompute aggression factors
        aggression = {}
        stage_multiplier = {'early': 0.5, 'middle': 2, 'late': 1.25}
        for color in [chess.WHITE, chess.BLACK]:
            aggression[color] = min(material[color] / (2 * material[not color]), 1.5) ** 2
            aggression[color] *= stage_multiplier[stage]

        # Precompute attackers for all squares using bitboard operations
        attackers_cache = {chess.WHITE: {}, chess.BLACK: {}}
        
        # Get all pieces for both colors
        white_pieces = board.occupied_co[chess.WHITE]
        black_pieces = board.occupied_co[chess.BLACK]
        
        # Precompute piece weights and space weights as constants
        piece_weight = {
            chess.PAWN: 0.5,
            chess.KNIGHT: 1.5,
            chess.BISHOP: 1.7,
            chess.ROOK: 1.2,
            chess.QUEEN: 1.0,
            chess.KING: 0.3
        }

        # Optimized attackers cache computation
        for square in chess.SQUARES:
            # Use bitwise operations for efficiency
            white_attackers = board.attackers_mask(chess.WHITE, square) & black_pieces
            black_attackers = board.attackers_mask(chess.BLACK, square) & white_pieces
            
            attackers_cache[chess.WHITE][square] = chess.SquareSet(white_attackers)
            attackers_cache[chess.BLACK][square] = chess.SquareSet(black_attackers)

        def piece_loop(color: chess.Color) -> dict:
            """Optimized piece evaluation using precomputed values and efficient bitboard operations."""
            
            # Precompute constants outside the loop
            enemy_color = not color
            king_square = KINGS[enemy_color]
            
            # Cache frequently accessed values
            material_values = self.MATERIAL
            heatmap_data = self.HEATMAP[stage]
            
            # Initialize scores
            scores = {
                'mobility': 0.0,
                'heatmap': 0.0,
                'development': 0.0,
                'aggression': 0.0,
                'space': 0.0,
                'file_control': 0.0,
                'outpost': 0.0,
                'coordination': 0.0
            }
            
            # Cache starting squares for development bonus
            starting_squares = {
                chess.WHITE: {
                    chess.KNIGHT: {chess.B1, chess.G1},
                    chess.BISHOP: {chess.C1, chess.F1}
                },
                chess.BLACK: {
                    chess.KNIGHT: {chess.B8, chess.G8},
                    chess.BISHOP: {chess.C8, chess.F8}
                }
            }[color]
            
            # Precompute pawn bitboards for file analysis
            all_pawns = board.pawns
            color_pawns = all_pawns & board.occupied_co[color]
            
            # Cache piece weights
            piece_weights = {
                chess.PAWN: 0.5,
                chess.KNIGHT: 1.5,
                chess.BISHOP: 1.7,
                chess.ROOK: 1.2,
                chess.QUEEN: 1.0,
                chess.KING: 0.3
            }
            
            space_weights = {
                chess.PAWN: 2.0,
                chess.KNIGHT: 1.5,
                chess.BISHOP: 1.0,
                chess.QUEEN: 1.0
            }
            
            # Initialize attack tracking
            attacks_cache = {}
            attacked_squares = {}
            
            # Process all pieces of the given color
            for square, piece in PIECES[color].items():
                if piece is None:
                    continue
                    
                piece_type = piece.piece_type
                piece_symbol = piece.symbol().upper()
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Calculate adjusted rank for heatmap
                heatmap_rank = 7 - rank if color == chess.WHITE else rank
                
                # Mobility calculation using efficient bitboard operations
                attacks = board.attacks(square)
                attack_count = len(attacks)
                attacks_cache[square] = attacks
                
                # Update attacked squares count
                for attacked_sq in attacks:
                    attacked_squares[attacked_sq] = attacked_squares.get(attacked_sq, 0) + 1
                
                # Weighted mobility score
                weight = piece_weights.get(piece_type, 1.0)
                scores['mobility'] += (attack_count ** 0.75) * weight * 0.5
                
                # Heatmap score with bounds checking
                if piece_symbol in heatmap_data and 0 <= heatmap_rank < 8 and 0 <= file < 8:
                    scores['heatmap'] += heatmap_data[piece_symbol][heatmap_rank][file] * 2.5
                
                # Development bonus for knights and bishops
                if piece_type in {chess.KNIGHT, chess.BISHOP} and square not in starting_squares[piece_type]:
                    scores['development'] += 1.5
                
                # Aggression score based on distance to enemy king
                if king_square is not None:
                    distance = chess.square_distance(square, king_square)
                    scores['aggression'] += material_values[piece_type] / max(1, distance) * 5
                
                # Space score based on rank advancement
                space_bonus = rank / 7.0 if color == chess.WHITE else (7 - rank) / 7.0
                space_weight = space_weights.get(piece_type, 0)
                scores['space'] += space_bonus * space_weight * 2.0
                
                # File control for rooks and queens
                if piece_type in {chess.ROOK, chess.QUEEN}:
                    file_mask = chess.BB_FILES[file]
                    
                    # Check if file is open or semi-open
                    file_pawns = file_mask & all_pawns
                    color_file_pawns = file_mask & color_pawns
                    
                    is_open = file_pawns == 0
                    is_semi_open = color_file_pawns == 0
                    
                    if is_open:
                        scores['file_control'] += 3.0
                    elif is_semi_open:
                        scores['file_control'] += 2.0
                    
                    # Bonus for rooks on 7th/2nd rank
                    if piece_type == chess.ROOK:
                        if (color == chess.WHITE and rank in {6, 7}) or (color == chess.BLACK and rank in {0, 1}):
                            rank_bonus = 3.5 if rank == 7 else 2.5
                            scores['file_control'] += rank_bonus
                
                # Outpost evaluation for knights and bishops
                if piece_type in {chess.KNIGHT, chess.BISHOP}:
                    # Check if piece is on opponent's side
                    is_on_enemy_side = (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3)
                    
                    if is_on_enemy_side:
                        # Check pawn protection
                        shield_rank = rank - 1 if color == chess.WHITE else rank + 1
                        shield_square = chess.square(file, shield_rank)
                        
                        pawn_shield = (0 <= shield_square < 64 and 
                                     board.piece_type_at(shield_square) == chess.PAWN and
                                     board.color_at(shield_square) == color)
                        
                        # Check if safe from enemy pawn attacks
                        is_safe = True
                        pawn_attack_offsets = [-7, -9] if color == chess.WHITE else [7, 9]
                        
                        for offset in pawn_attack_offsets:
                            attack_square = square + offset
                            if 0 <= attack_square < 64:
                                attack_file = chess.square_file(attack_square)
                                if abs(attack_file - file) == 1:
                                    attack_piece = board.piece_type_at(attack_square)
                                    attack_color = board.color_at(attack_square)
                                    if attack_piece == chess.PAWN and attack_color == enemy_color:
                                        is_safe = False
                                        break
                        
                        if pawn_shield and is_safe:
                            scores['outpost'] += 5.0 if piece_type == chess.KNIGHT else 3.5
                        elif is_safe:
                            scores['outpost'] += 2.5 if piece_type == chess.KNIGHT else 1.5
                
                # Coordination score
                if piece_type != chess.KING:
                    # Count friendly pieces protecting this square
                    protectors = attackers_cache[color][square]
                    protector_count = len(protectors)
                    scores['coordination'] += protector_count * 0.75
                    
                    # Penalty for undefended pieces
                    attackers = attackers_cache[enemy_color][square]
                    attacker_count = len(attackers)
                    if attacker_count > 0 and protector_count == 0:
                        scores['coordination'] -= 3.0
            
            return {
                "mobility_score": scores['mobility'],
                "heatmap_score": scores['heatmap'],
                "minor_piece_development_bonus": scores['development'],
                "aggression_score": scores['aggression'],
                "attacks_cache": attacks_cache,
                "space_score": scores['space'],
                "attacked_squares": attacked_squares,
                "file_score": scores['file_control'],
                "outpost_score": scores['outpost'],
                "base_coordination_score": scores['coordination']
            }

        def evaluate_player(color: chess.Color) -> float:

            king_square = KINGS[color]
            enemy_king_square = KINGS[not color]
            if king_square is None or enemy_king_square is None:
                return 0

            PIECE_SCORES = piece_loop(color)

            # Cache attacked squares and attacks per piece to avoid repeated calls
            attacks_cache = PIECE_SCORES["attacks_cache"]
            attacked_squares = [sq for attacks in attacks_cache.values() for sq in attacks]

            # Legal moves are only important if it's the player's turn
            legal_moves = list(board.legal_moves) if color == board.turn else []

            def coverage() -> float:
                attack_bonus = 0
                cover_bonus = 0

                attacked_square_counts = {}
                center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
                
                # Calculate cover bonus denominator once
                cover_denominator = sum(self.MATERIAL[piece_map[sq].piece_type] for sq in attacks_cache.keys())
                cover_bonus = len(attacked_squares) / cover_denominator if cover_denominator > 0 else 0

                # Single loop through attacked squares to build counts
                for sq in attacked_squares:
                    attacked_square_counts[sq] = attacked_square_counts.get(sq, 0) + 1

                # Single loop through unique squares
                for square, count in attacked_square_counts.items():
                    attack_bonus += count ** 1.25

                    piece_type = board.piece_type_at(square)
                    piece_color = board.color_at(square)
                    rank = chess.square_rank(square)

                    # Pre-calculate aggression multiplier
                    agg_mult = aggression[color]
                    
                    # Center control bonus
                    if square in center_squares:
                        attack_bonus += 2.5 * agg_mult
                    
                    # Enemy half bonus
                    if (color == chess.WHITE and rank > 3) or (color == chess.BLACK and rank < 4):
                        attack_bonus += 1.5 * agg_mult

                    if piece_type is not None:
                        piece_weight_val = piece_weight[piece_type] ** 0.6
                        stage_mult = 7 if stage == 'late' else 4
                        back_rank_mult = 6 if stage == 'late' else 3
                        
                    # Enemy half piece bonus - increased for active pieces
                    if piece_type is not None:
                        piece_weight_val = piece_weight.get(piece_type, 1.0) ** 0.6
                        stage_mult = 7 if stage == 'late' else 4
                        back_rank_mult = 6 if stage == 'late' else 3
                        
                        if (color == chess.WHITE and rank > 3) or (color == chess.BLACK and rank < 4):
                            attack_bonus += piece_weight_val * agg_mult * stage_mult * 1.5
                            
                        # Back rank penalty - increased to discourage passive pieces
                        if (color == chess.WHITE and rank < 2) or (color == chess.BLACK and rank > 5):
                            attack_bonus -= piece_weight_val * agg_mult * back_rank_mult * 1.3
                            
                        # Attack/defense bonuses - enhanced for opponent piece activity reduction
                        if piece_color != color and piece_type in self.MATERIAL:
                            attack_bonus += self.MATERIAL[piece_type] ** 2.0 * agg_mult * 1.5
                        elif piece_type in self.MATERIAL:
                            attack_bonus += self.MATERIAL[piece_type] ** 1.1 * 2.5 * aggression[not color]

                # Handle edge case where attack_bonus is 0
                if attack_bonus == 0:
                    return cover_bonus
                
                return abs(attack_bonus) ** 0.5 * (1 if attack_bonus > 0 else -1) + cover_bonus

            def control() -> float:
                control_bonus = 0

                # Encourage balanced control
                len_attacks = [len(attacks) for square, attacks in attacks_cache.items() if board.piece_type_at(square) not in [chess.PAWN, chess.KING]]
                variance = sum((x - sum(len_attacks) / len(len_attacks)) ** 2 for x in len_attacks)
                control_bonus -= variance * 0.5

                for square, attacks in attacks_cache.items():
                    control_bonus += len(attacks) ** 0.35

                    if square not in [chess.E4, chess.E5, chess.D4, chess.D5]:
                        continue
                    piece = piece_map.get(square)
                    if piece is None:
                        continue
                    control_bonus += self.MATERIAL[piece.piece_type]
                    if piece.piece_type == chess.PAWN:
                        control_bonus += 4
                    elif piece.piece_type == chess.KNIGHT:
                        control_bonus += 2

                return control_bonus

            def material_score() -> float:
                base_score = material[color] ** 2 / (material[not color] + 1)
                return base_score + PIECE_SCORES["mobility_score"]

            def heatmap() -> float:
                return PIECE_SCORES["heatmap_score"]

            def low_legal_penalty() -> float:
                if color == board.turn:
                    legal_moves_local = legal_moves
                else:
                    board.turn = color
                    legal_moves_local = list(board.legal_moves)
                    board.turn = not color

                return 1 / len(legal_moves_local) if legal_moves_local else 1

            def minor_piece_bonus() -> float:
                return PIECE_SCORES["minor_piece_development_bonus"]

            def king_safety_penalty() -> float:
                king_penalty = 0
                king_start_square = chess.E1 if color == chess.WHITE else chess.E8
                has_moved = king_square != king_start_square

                if stage != 'late' and (not board.has_castling_rights(color)) and has_moved:
                    king_penalty += 50.0
                king_moves = list(attacks_cache.get(king_square, []))
                king_penalty -= len(king_moves) ** 0.5

                # Treat the king as a queen and penalise for the moves that the queen can make
                attacks = chess.BB_DIAG_ATTACKS[king_square][chess.BB_DIAG_MASKS[king_square] & board.occupied]
                attacks |= (chess.BB_RANK_ATTACKS[king_square][chess.BB_RANK_MASKS[king_square] & board.occupied] |
                                chess.BB_FILE_ATTACKS[king_square][chess.BB_FILE_MASKS[king_square] & board.occupied])
            
                # Get number of attacks
                num_attacks = bin(attacks).count('1')
                
                king_penalty += num_attacks ** 2 * 0.5
                
                return king_penalty

            def pawn_structure() -> float:
                pawn_score = 0

                pawns_bb = white_pawns if color == chess.WHITE else black_pawns
                enemy_pawns_bb = black_pawns if color == chess.WHITE else white_pawns

                pawn_files_bb = [chess.BB_FILES[file] & pawns_bb for file in range(8)]

                for file_bb in pawn_files_bb:
                    pawn_count = bin(file_bb).count('1')
                    pawn_score -= 1.5 * max(0, pawn_count - 1) ** 2

                # Isolated pawns
                for file in range(8):
                    if pawn_files_bb[file]:
                        left = pawn_files_bb[file - 1] if file > 0 else 0
                        right = pawn_files_bb[file + 1] if file < 7 else 0
                        if not (left or right):
                            pawn_score -= 1.5

                # Connected pawns
                if color == chess.WHITE:
                    connected = ((pawns_bb << 7) | (pawns_bb << 9)) & pawns_bb
                else:
                    connected = ((pawns_bb >> 7) | (pawns_bb >> 9)) & pawns_bb
                if connected:
                    pawn_score += 1.0

                # Passed pawns
                for square in chess.SquareSet(pawns_bb):
                    file = chess.square_file(square)
                    rank = chess.square_rank(square)
                    if color == chess.WHITE:
                        if rank < 7:
                            front_span = chess.BB_RANKS[rank+1]
                            for r in range(rank+2, 8):
                                front_span |= chess.BB_RANKS[r]
                        else:
                            front_span = 0
                        front_mask = 0
                        for f in range(max(0, file-1), min(7, file+1)+1):
                            front_mask |= chess.BB_FILES[f]
                        front_mask &= front_span
                        if enemy_pawns_bb & front_mask == 0:
                            pawn_score += 2.0
                            if connected & (1 << square):
                                pawn_score += 1.0
                    else:
                        if rank > 0:
                            front_span = chess.BB_RANKS[rank-1]
                            for r in range(rank-2, -1, -1):
                                front_span |= chess.BB_RANKS[r]
                        else:
                            front_span = 0
                        front_mask = 0
                        for f in range(max(0, file-1), min(7, file+1)+1):
                            front_mask |= chess.BB_FILES[f]
                        front_mask &= front_span

                        # Reward passed pawns
                        if enemy_pawns_bb & front_mask == 0:
                            rank_bonus = (1 + (rank / 7 if color == chess.WHITE else 1 - rank / 7)) ** 4 * 5
                            stage_bonus = 4 if stage == 'late' else 1
                            pawn_score += (3.0 if stage == 'late' else 2.0) * rank_bonus * stage_bonus
                            if connected & (1 << square):
                                pawn_score += (2.0 if stage == 'late' else 1.0) * rank_bonus * stage_bonus # Extra bonus if pawn is connected

                    # Give a small penalty based on their distance from the promotion rank
                    prom_rank = 7 if color == chess.WHITE else 0
                    pawn_score -= abs(prom_rank - rank) ** 1.5 * (0.75 if stage == 'late' else 0.25)

                if king_square is not None:
                    for square in chess.SquareSet(pawns_bb):
                        rank = chess.square_rank(square)
                        if chess.square_distance(square, king_square) <= 1 and ((color == chess.WHITE and rank > chess.square_rank(king_square)) or (color == chess.BLACK and rank < chess.square_rank(king_square))):
                            pawn_score += 1.5 if stage != "late" else 0.5

                return pawn_score

            def attack_quality() -> float:
                aggression_score = PIECE_SCORES["aggression_score"]
                material_diff = material[color] - material[not color]

                if board.is_check():
                    if board.turn == color:
                        aggression_score -= 3
                    else:
                        aggression_score += 3

                for move in legal_moves:

                    # Reward types of moves
                    if board.is_capture(move):
                        aggression_score += 0.75
                    else: # No victim, so continue
                        continue

                    # Reward moves that attack opponent's pieces
                    victim = board.piece_type_at(move.to_square)
                    attacker = board.piece_type_at(move.from_square)
                    
                    # Skip if no victim or attacker (shouldn't happen, but it stops the errors showing up)
                    if not victim or not attacker:
                        continue

                    victim_material = self.MATERIAL[victim]
                    attacker_material = self.MATERIAL[attacker]
                    if victim_material > attacker_material:
                        aggression_score += 3.0
                    if material_diff + victim_material - attacker_material > 1.5:
                        aggression_score += 2.5
                    elif material_diff + victim_material - attacker_material < 0.75:
                        aggression_score -= 4.0

                for square, piece in PIECES[not color].items():
                    attackers = attackers_cache[color][square]
                    defenders = attackers_cache[not color][square]

                    if attackers:
                        if len(defenders) == 0:
                            aggression_score += self.MATERIAL[piece.piece_type] * len(attackers) * 2.0
                        else:
                            defender = board.piece_type_at(next(iter(defenders)))
                            if defender is None:
                                continue
                            defender_value = self.MATERIAL[defender]
                            if self.MATERIAL[piece.piece_type] < defender_value:
                                aggression_score += self.MATERIAL[piece.piece_type] * len(attackers)
                            else:
                                aggression_score -= self.MATERIAL[piece.piece_type] * len(attackers) * 0.5

                return aggression_score

            def piece_coordination() -> float:
                """Evaluate how well pieces work together."""
                coordination_score = PIECE_SCORES["base_coordination_score"]
                attacked_squares = PIECE_SCORES["attacked_squares"]
                
                # Multiplier for coordinated attacks
                for count in attacked_squares.values():
                    if count >= 2:
                        coordination_score += count * 1.5
                
                return coordination_score

            def outpost_evaluation() -> float:
                """Evaluate knight and bishop outposts."""
                return PIECE_SCORES["outpost_score"]

            def open_file_evaluation() -> float:
                """Evaluate control of open and semi-open files."""
                return PIECE_SCORES["file_score"]

            score = 0
            score -= king_safety_penalty() * (0.75 if stage == 'late' else 7.5)
            score -= (4 * low_legal_penalty()) ** 1.5 * (aggression[not color] ** 2)
            score += material_score() ** 2.75 * (35 if stage == 'late' else 10)
            score += coverage() * aggression[color] * (0.55 if stage == 'late' else 0.2)
            score += cse(heatmap(), (2 if stage == 'early' else 1.75 if stage == 'late' else 1)) * aggression[not color] * (15 if stage == 'late' else 7 if stage == 'early' else 5)
            score += cse(control(), 1.25) * aggression[color] * (1.2 if stage == 'late' else 0.65 if stage == 'early' else 0.45)
            score += minor_piece_bonus() * 15 * aggression[color]
            score += cse(pawn_structure(), 1.5) * (10 if stage == 'late' else 3.75)
            score += attack_quality() ** 1.3 * aggression[color] * (2 if stage == 'late' else 5)
            score += piece_coordination() * aggression[color] * (0.25 if stage == 'late' else 0.1)
            score += outpost_evaluation() * aggression[color] * (0.25 if stage == 'late' else 0.1)
            score += open_file_evaluation() * aggression[color] * (0.25 if stage == 'late' else 0.1)

            if isinstance(score, complex):
                print("\nAGG", aggression[color])
                print("EAG", aggression[not color])
                print("-KSP", king_safety_penalty() * 5.5)
                print("-LLP", (4 * low_legal_penalty()) ** 1.5 * (aggression[not color] ** 2))
                print("+MS", material_score() ** 2 * 20)
                print("+COV", coverage() * 0.1 * aggression[color])
                print("+HEAT", cse(heatmap(), (2 if stage == 'early' else 1.5 if stage == 'late' else 1)) * aggression[not color] * (10 if stage == 'late' else 7 if stage == 'early' else 5))
                print("+CTRL", (control() ** 1.25) * 0.35 * aggression[color] * (2 if stage == 'late' else 1.5 if stage == 'early' else 1))
                print("+MIN", minor_piece_bonus() * 15 * aggression[color])
                print("+PAWN", cse(pawn_structure(), 7/5) * 2.5 * (2 if stage == 'late' else 1.5))
                print("+ATT", attack_quality() ** 1.3 * aggression[color] * 15)
                raise ValueError("Score is complex")

            return score

        score = 0
        score += evaluate_player(chess.WHITE)
        score -= evaluate_player(chess.BLACK)
        self.save_evaluation(board, score, 0)

        return score
    
    def instant_response(self, board: chess.Board) -> chess.Move | None:
        """
        Determine an instant move for the computer player.

        The method attempts to quickly return a move by checking for predefined 
        strategies and stored winning moves:
        1. It first attempts a random opening move.
        2. If no opening move is available, it retrieves a move from the Syzygy 
        endgame tablebases if possible. (MOVED TO BEST_MOVE)
        3. Finally, it checks for a stored winning move for the current board 
        position.

        :param board: The current state of the board
        :type board: chess.Board
        :return: The selected move, or None if no move is available
        :rtype: chess.Move | None
        """

        # First try a random opening move
        opening_best = self.random_opening_move(board)
        if opening_best is not None:
            print("Using random opening move")
            self.conn.close()
            return opening_best

        # Check if there is a stored winning move for the current position
        stored_move = self.get_stored_winning_move(board)
        if stored_move is not None and stored_move in board.legal_moves:
            print("Using stored winning move")
            self.conn.close()
            return stored_move
    
        return None

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

        self.conn = sqlite3.connect(self.TRANSPOSITION_PATH)
        self.cursor = self.conn.cursor()

        instant_response = self.instant_response(board)
        if instant_response is not None:
            return instant_response
        
        ####################################################################################################

        print(f"{"white" if board.turn == chess.WHITE else "black"} move")

        self.start_time = time.time()
        self.timeout = timeout

        depth = 1
        best_move = None
        reset = False

        # Reset metrics
        self.nodes_explored = 0
        self.leaf_nodes_explored = 0
        self.alpha_cuts = 0
        self.beta_cuts = 0
        self.prunes = 0
        self.cache_hits = 0
        
        maxmin = max if board.turn == chess.WHITE else min

        moves = list(board.legal_moves)  # Convert generator to list for membership checks

        move_score_map: list[tuple[chess.Move, float]] = [(move, 0) for move in moves] # Initialize once before loop

        # Sort by the MVV-LVA heuristic
        move_score_map.sort(key=lambda x: self.mvv_lva_score(board, x[0]), reverse=True)

        stage = self.get_game_stage(board.piece_map())

        while not self.is_timeup():

            print(f"""DEPTH {depth}: """,end='\t')

            move_score_map.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
            moves = [move for move, _ in move_score_map]

            # Get best score: If can_sygyzy and score is not high enough (or is not completely lost) after a shallow search, then use Sygyzy
            if depth > 3:
                best_score = maxmin(move_score_map, key=lambda x: x[1])[1]
                move = self.best_sygyzy(board, best_score)
                if move is not None:
                    print("Using Syzygy:",board.san(move))
                    return move

            # Gradually filter out based on the previous scores
            if depth > 3 and not reset:
                if stage != 'late' and depth == 3:
                    threshold = 0.1 * (len(list(board.legal_moves)) / len(moves))
                else:
                    threshold = 0.15 * (len(list(board.legal_moves)) / len(moves))
                print("Threshold:",threshold)
                turning_point = self._turning_point([score for _, score in move_score_map], threshold=threshold)
                move_score_map = move_score_map[:turning_point]
                moves = moves[:turning_point]
            print(len(moves),"moves:",[board.san(m) for m in moves])

            # If only one move left, return it
            if len(moves) == 1:
                self.display_metrics()
                self.print_board_feedback(board, move_score_map)
                return moves[0]

            # Create a dictionary for quick lookup and update of scores
            move_score_dict = dict(move_score_map)

            current_best_move = None
            current_best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

            # Evaluate each mmove
            for i, move in enumerate(moves):

                # LMR : reduce depth for later moves in the list
                reduction = 0
                if depth > 3 and stage != 'late': # Only apply when the computer has a good idea of the position and not in the endgame
                    if i > self._turning_point([score for _, score in move_score_map], threshold=1.5 - (i / len(moves))): # Dynamic threshold based on move number
                        print("LMR",end='\t')
                        reduction += 1

                board.push(move)
                # Check if the move has already been evaluated
                score = self.evaluate_from_db(board, depth)

                if score is not None:
                    move_score_dict[move] = score
                else:
                    score = self.minimax(board, depth - reduction, float('-inf'), float('inf'), original_depth=depth - reduction, heuristic_eliminate=False, use_mvv_lva=True)
                board.pop()

                if self.is_timeup():
                    print("TIMEUP")
                    break

                # Normalise scores as they grow extremely large
                print(f"{board.san(move)} : {self.normalise_score(score)}",end='\t',flush=True)
                
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

                # Remove moves that lead to checkmate
                if score == self.WORST_SCORE:
                    del move_score_dict[move]

                if _should_terminate(list(move_score_dict.items())):
                    print("TERMINATED EARLY")
                    break
            
            print()

            # If all moves lead to checkmate, add every move back into the scope to be searched
            if not move_score_dict and not reset:
                print("All moves lead to checkmate, resetting move scope to all legal moves")
                # Reset move_score_map to include all legal moves with initial score 0
                move_score_map = [(move, 0) for move in board.legal_moves]
                depth = 0
                reset = True
                continue
            elif not move_score_dict:
                self.conn.close()
                return best_move

            # Update move_score_map from the dictionary for next iteration
            move_score_map = list(move_score_dict.items())

            # Terminate early if an immediate win is found
            if _should_terminate(move_score_map):

                # Choose best move from current depth
                best_move = current_best_move if current_best_move is not None else maxmin(move_score_map, key=lambda x: x[1])[0]

                # Save the winning move and the position before the move
                self.save_winning_move(board, best_move)
                break

            best_move = current_best_move if current_best_move is not None else maxmin(move_score_map, key=lambda x: x[1])[0]

            print("BEST:",board.san(best_move))

            depth += 1

            self.conn.commit()

        self.conn.close()
        self.display_metrics()
        self.print_board_feedback(board, move_score_map)

        return best_move

    ##################################################
    #                     EXTRAS                     #
    ##################################################

    def get_game_stage(self, piece_map: dict) -> str:
        """Return the current stage of the game."""

        num_pieces = len([piece for piece in piece_map.values() if piece.piece_type != chess.PAWN])
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

    def normalise_score(self, score: float) -> float:
        return round(score / self.ESTIMATED_PAWN_VALUE, 2)
    
    def display_metrics(self) -> None:
        
        elapsed = time.time() - self.start_time if self.start_time is not None else float("nan")
        print(f"""
        Nodes explored : {self.nodes_explored} | NPS : {self.nodes_explored / elapsed:.2f}
        Leaf nodes explored : {self.leaf_nodes_explored} | LNPS : {self.leaf_nodes_explored / elapsed:.2f}
        Alpha cutoffs : {self.alpha_cuts}
        Beta cutoffs : {self.beta_cuts}
        Total prunes : {self.prunes}
        Cache hits : {self.cache_hits}
        """)

    @classmethod
    def estimate_pawn_value(cls, depth: int) -> None:
        """
        Estimate the value of a pawn based on the current position, iterating depths up to the depth specified.

        :param depth: The max depth to search in the Minimax algorithm
        :type depth: int
        :return: The estimated value of a pawn
        """

        normal_fen = chess.STARTING_FEN
        fen_without_pawn = "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"

        # Create temporary computer instance for evaluation
        c = Computer(chess.WHITE)
        
        for i in range(depth):

            print("DEPTH",i + 1)

            normal_score = c.minimax(chess.Board(normal_fen), i + 1, float('-inf'), float('inf'), original_depth=i + 1, heuristic_sort=False, heuristic_eliminate=False, use_mvv_lva=True)
            no_pawn_score = c.minimax(chess.Board(fen_without_pawn), i + 1, float('-inf'), float('inf'), original_depth=i + 1, heuristic_sort=False, heuristic_eliminate=False, use_mvv_lva=True)

            c.display_metrics()
            print("Normal score:", normal_score)
            print("No pawn score:", no_pawn_score)
            print(normal_score - no_pawn_score)
                            
    def print_board_feedback(self, board: chess.Board, scoremap: list[tuple[chess.Move, float]]) -> None:
        """
        Display information based on what the computer makes of the position, including:
        1. The top 3 moves and their scores.
        2. What it thinks the evaluation of the position is.

        :param board: The current state of the board
        :type board: chess.Board
        :param scoremap: A list of tuples containing moves and their scores
        :type scoremap: list[tuple[chess.Move, float]]
        """

        scoremap.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)

        position_responses: dict[float | int, str] = {
            -3 : "I should resign!",
            -2 : "I am losing, but not as badly as I could be.",
            -1 : "I am losing, but I think I can still turn it around.",
            1 : "the position is equal.",
            2 : "I have a slight advantage.",
            3 : "I have a good advantage.",
            float('inf') : "I have a huge advantage!"
        }

        score = self.normalise_score(scoremap[0][1])
        score = score if board.turn == chess.WHITE else -score

        response = "the response is broken. :("
        for s, response in position_responses.items():
            if score <= s:
                break

        print(f"""
I evaluate the position at {score} in my favour.
I think that {response}
""")


def profile_evaluation() -> None:

    import cProfile

    c = Computer(chess.WHITE)

    def generate_positions(n: int=100, moves: int=10):
        for _ in range(n):
            board = chess.Board()
            for _ in range(moves):
                if not board.is_game_over():
                    board.push(rnd.choice(list(board.legal_moves)))
            yield board

    print("Generating...")
    boards = list(generate_positions(n=1000, moves=10))
    print("Generated.")

    print("Warming up the interpreter...")
    for _ in range(int(3e4)):
        _ = 12345 ** 2345
    print("Interpreter warmed up.")

    def evaluate_boards() -> None:
        for board in boards:
            c.evaluate(board)

    cProfile.runctx('evaluate_boards()', globals(), locals(), sort='cumtime')

def main():

    FEN = chess.STARTING_FEN
    # FEN = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kk - 0 1" # Test opening book out of theory
    FEN = "3n4/8/5k2/8/4p3/1p2B3/1P2BPK1/8 w - - 0 74" # Test cleanup ability when clearly winning

    board = chess.Board(FEN)
    players = [Computer(board.turn), Computer(not board.turn)]

    while not board.is_game_over():
        print(board,"\n\n")
        player = players[0] if board.turn == chess.WHITE else players[1]
        move = player.best_move(board, timeout=15)
        if move is None:
            break
        print("\n\nMove:", board.san(move))
        board.push(move)
    print(board)
    print("GAME OVER!")

if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run('main()',sort='cumulative')
    # profile_evaluation()
    # Computer.estimate_pawn_value(5)
