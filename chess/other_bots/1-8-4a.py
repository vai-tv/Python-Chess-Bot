"""
Formulae can be found here: https://www.desmos.com/calculator/hbrvuxpnqq
"""

import chess
import collections
import math
import multiprocessing as mp
import random as rnd
import requests
import sys
import time
import urllib.parse

from typing import Hashable

sys.setrecursionlimit(int(1e6))

__version__ = '1.8.4a'
NAME = 'XXIEvo'

class Computer:

    BEST_CPU_COUNT = mp.cpu_count() #math.floor(mp.cpu_count() * 0.875)

    def __init__(self, color: chess.Color, workers: int=BEST_CPU_COUNT, 
                 shared_transposition_table: dict[Hashable, float] | None = None,
                 shared_pawn_hash_cache: dict[tuple[int, int], float] | None = None):
        self.color = color
        self.workers = min(workers, mp.cpu_count())
        self.BEST_SCORE = float('inf') if color == chess.WHITE else float('-inf')
        self.WORST_SCORE = float('-inf') if color == chess.WHITE else float('inf')
        self.MAXMIN = max if color == chess.WHITE else min

        self.timeout: float | None = None
        self.start_time: float | None = None

        # Initialize killer moves dictionary: depth -> list of killer moves
        self.killer_moves: dict[int, list[chess.Move]] = {}

        # Metrics
        self.nodes_explored = 0
        self.leaf_nodes_explored = 0
        self.quiescence_nodes_explored = 0
        self.alpha_cuts = 0
        self.beta_cuts = 0
        self.prunes = 0
        self.cache_hits = 0

        # Evaluation caches - use shared caches if provided, otherwise create local ones
        self.transposition_table: dict[Hashable, float] = shared_transposition_table if shared_transposition_table is not None else {}
        self.pawn_hash_cache: dict[tuple[int, int], float] = shared_pawn_hash_cache if shared_pawn_hash_cache is not None else {}
        
        # History heuristic table
        self.history_table: dict[chess.Move, int] = {}

    ##################################################
    #            OPENING BOOKS AND syzygy            #
    ##################################################

    SYZYGY_URL = "https://tablebase.lichess.ovh/standard?fen="
    OPENING_URL = "https://explorer.lichess.ovh/master?fen="

    OPENING_LEAVE_CHANCE = 0.05  # Chance to leave the opening book
    MIN_OPENING_GAMES = 100 # Only consider opening moves with at least this number of games in the book

    def can_syzygy(self, board: chess.Board) -> bool:
        num_pieces = chess.popcount(board.occupied)
        if num_pieces > 7:
            return False
        return True

    def syzygy_query(self, board: chess.Board) -> dict:
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

        url = self.SYZYGY_URL + fen_encoded

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            time.sleep(3)
            return self.syzygy_query(board)
        else:
            raise requests.RequestException(f"Request to Syzygy tablebase server failed with status code {response.status_code}")

    def syzygy_score(self, board: chess.Board) -> float:
        ismaximising = board.turn == chess.WHITE
        category = self.syzygy_query(board)["category"]
        if category == "win":
            return float('inf') if ismaximising else float('-inf')
        elif category == "loss":
            return float('-inf') if ismaximising else float('inf')
        else:
            return 0
    
    def best_syzygy(self, board: chess.Board) -> chess.Move | None:
        """
        Get the best move from the Syzygy tablebase server for the given board position.
        Chooses moves optimally:
            - Win  → fastest mate (lowest DTZ)
            - Draw → safest draw (highest DTZ)
            - Loss → longest survival (highest DTZ)
        """
        if not self.can_syzygy(board):
            return None

        response = self.syzygy_query(board)
        category = response["category"]
        moves = response.get("moves", [])
        if not moves:
            return None

        if category == "win":
            # Pick move with smallest DTZ (fastest win)
            best_move = min(moves, key=lambda m: m.get("dtz", 0))
        elif category == "draw":
            # Pick safest draw (maximize DTZ)
            best_move = max(moves, key=lambda m: m.get("dtz", 0))
        else:  # "loss"
            # Delay defeat as much as possible
            best_move = max(moves, key=lambda m: m.get("dtz", 0))

        return chess.Move.from_uci(best_move["uci"])

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

        if self.get_game_stage(board.piece_map()) == Computer.stageLATE:
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
            color = "white" if board.turn == chess.WHITE else "black"

            # Only choose weights with over 1000 games to prevent bad moves
            weights = [move[color] for move in moves if move[color] > self.MIN_OPENING_GAMES] 
            moves = [move for move in moves if move[color] > self.MIN_OPENING_GAMES]
            if not weights:
                return None
            chosen_move = rnd.choices(moves, weights=weights, k=1)[0]
            return chess.Move.from_uci(chosen_move["uci"])
        return None

    ##################################################
    #                   HEURISTICS                   #
    ##################################################
    
    MIN_CUTOFF = 5
    MIN_CUTOFF_DEPTH = 3

    def _turning_point(self, prenormal_scores: list[float], threshold: float=0.25, depth: int=0) -> int:
        """
        Find the optimal turning point in sorted scores using median gap size.
        
        The method identifies moves that are significantly worse than the best move
        by calculating the median gap between consecutive scores and cutting where
        the gap exceeds the median.
        
        :param prenormal_scores: A sorted list of scores (descending for white, ascending for black)
        :param threshold: Base threshold for quality drop (0-1, higher = keep more moves)
        :param depth: Depth of recursion
        :return: The index of the turning point or len(scores) if no significant gap is found
        """
        
        if not prenormal_scores:
            return -1
        if len(prenormal_scores) == 1:
            return 1
        
        # Normalise all scores
        scores: list[float] = [self.normalise_score(score) for score in prenormal_scores]
            
        # Get the best score (first element for white, last for black - but scores are pre-sorted)
        best_score = scores[0] if scores[0] > scores[-1] else scores[-1]
        
        # Calculate score range in centipawns for better understanding
        score_range = abs(max(scores) - min(scores))

        if depth < self.MIN_CUTOFF_DEPTH:
            # Return the larget gap that is at least 750 centipawns
            for i in range(1, len(scores)):
                gap = abs(scores[i] - scores[i-1])
                if gap > 7.5:
                    return i
            return len(scores)
        
        # If all moves are very close in value, keep them all
        if score_range < 3:  # Less than 3 centipawns difference
            return len(scores)
        
        # Calculate gaps between consecutive scores
        gaps = []
        for i in range(1, len(scores)):
            gap = abs(scores[i] - scores[i-1])
            # Immediately cut if gap is greater than 300 centipawns
            if gap > 3.0:
                return i

            gaps.append(gap)
        
        # If there are no gaps (only one score), keep all moves
        if not gaps:
            return len(scores)
        
        # Find median gap size
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        
        # Use median gap as threshold for significant quality drop
        # Apply threshold scaling to median gap
        scaled_median_gap = median_gap * (1.0 + threshold)
        
        # Always remove moves that are clearly bad (more than 500 centipawns worse than best)
        absolute_cutoff = 500.0
        absolute_min_score = best_score - absolute_cutoff if best_score == scores[0] else best_score + absolute_cutoff
        
        # Find the first move where the gap to the previous move exceeds the median gap
        # or the score falls below the absolute cutoff
        for i in range(self.MIN_CUTOFF, len(scores)):
            current_gap = abs(scores[i] - scores[i-1])
            
            if best_score == scores[0]:  # Descending order (white)
                if current_gap > scaled_median_gap or scores[i] < absolute_min_score:
                    return i
            else:  # Ascending order (black)
                if current_gap > scaled_median_gap or scores[i] > absolute_min_score:
                    return i
        
        # If no significant gaps found, keep all moves
        return len(scores)

    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> float:
        """
        Calculate the MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score for a move.

        :param board: The current state of the board
        :param move: The move to score
        :return: A float score for move ordering
        """
        victim = board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)

        if victim is None or attacker is None:
            return 0

        victim_value = self.MATERIAL[victim]
        attacker_value = self.MATERIAL[attacker]

        # Higher score for capturing more valuable victim with less valuable attacker
        return (victim_value * 10) - attacker_value

    def see_score(self, board: chess.Board, move: chess.Move) -> float:
        """
        Static Exchange Evaluation (SEE) score for a move.
        Returns the expected material gain from the exchange.
        
        :param board: The current state of the board
        :param move: The move to evaluate
        :return: SEE score (positive for good captures, negative for bad ones)
        """
        if not board.is_capture(move):
            return 0
            
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Get the pieces involved
        attacker = board.piece_type_at(from_sq)
        victim = board.piece_type_at(to_sq)
        
        if attacker is None or victim is None:
            return 0
            
        # Base gain is victim value minus attacker value
        gain = self.MATERIAL[victim] - self.MATERIAL[attacker]
        
        # Simple approximation - in a real implementation, you'd do a full SEE
        # For now, we'll use a simplified version that considers the piece values
        
        # Check if the capture is defended
        defenders = board.attackers(not board.turn, to_sq)
        attackers = board.attackers(board.turn, to_sq)
        
        # If there are more defenders than attackers, it might be a bad capture
        if len(defenders) > len(attackers):
            gain -= self.MATERIAL[attacker] * 0.5
            
        return gain

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

    def see_ordering(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Order moves using SEE (Static Exchange Evaluation) heuristic.

        :param board: The current state of the board
        :param moves: List of moves to order
        :return: List of moves ordered by SEE score descending
        """
        scored_moves = [(move, self.see_score(board, move)) for move in moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in scored_moves]

    def history_ordering(self, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Order moves using history heuristic.
        
        :param moves: List of moves to order
        :type moves: list[chess.Move]
        :return: List of moves ordered by history score descending
        :rtype: list[chess.Move]
        """
        scored_moves = [(move, self.history_table.get(move, 0)) for move in moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]


    ##################################################
    #                   EVALUATION                   #
    ##################################################

    stageEARLY = 1
    stageMIDDLE = 2
    stageLATE = 3
    stageLATELATE = 4

    HEATMAP: dict[int, dict[int, list[list[float]]]] = {
        stageEARLY: {
            1: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0], [1.0, 1.5, 2.0, 3.5, 3.5, 2.0, 1.5, 1.0], [0.0, -0.5, -0.25, 3.0, 3.0, -0.25, -0.5, 0.0], [0.0, -0.5, -1.0, -0.5, -0.5, -1.0, -0.5, 0.0], [1.0, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            2: [[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0], [-4.0, -2.0, -0.5, -0.5, -0.5, -0.5, -2.0, -4.0], [-3.0, 0.0, 0.5, 0.75, 0.75, 0.5, 0.0, -3.0], [-3.0, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -3.0], [-3.0, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -3.0], [-3.0, 0.0, 0.5, 0.75, 0.75, 0.5, 0.0, -3.0], [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0], [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]],
            3: [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [-1.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, -1.0], [-1.0, 0.125, 0.5, 0.5, 0.5, 0.5, 0.125, -1.0], [-1.0, 0.25, 0.875, 0.5, 0.5, 0.875, 0.25, -1.0], [-1.0, 0.375, 0.625, 0.75, 0.75, 0.625, 0.375, -1.0], [-1.5, 0.5, 0.125, 0.25, 0.25, 0.125, 0.5, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
            4: [[-0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.25], [0.5, 1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.75, 0.25, 0.375, 0.75, 0.75, 0.375, 0.25, -0.5], [-1.0, -0.75, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0]],
            5: [[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]],
            6: [[-4, -5, -5, -5, -5, -5, -5, -4], [-4, -5, -5, -5, -5, -5, -5, -4], [-3, -5, -5, -5, -5, -5, -5, -3], [-3, -4, -5, -5, -5, -5, -4, -3], [-2, -3, -4, -5, -5, -4, -3, -2], [-1, -2, -3, -4, -4, -3, -2, -1], [0, -1, -2, -3, -3, -2, -1, 0], [2, 3, 2, 0, 0, 2, 4, 2]]
        },
        stageMIDDLE: {
            1: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0], [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0], [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0], [0.5, 0.0, -0.5, 2.0, 2.0, -0.5, 0.0, 0.5], [0.5, 0.25, -1.0, 1.0, 1.0, -1.0, 0.25, 0.5], [1.0, 1.0, 1.0, -1.5, -1.5, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            2: [[-4.0, -3.0, -2.0, -2.0, -2.0, -2.0, -3.0, -4.0], [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0], [-2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, -2.0], [-2.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, -2.0], [-2.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, -2.0], [-2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, -2.0], [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0], [-4.0, -3.0, -2.0, -2.0, -2.0, -2.0, -3.0, -4.0]],
            3: [[-1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -1.0, -1.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -0.5], [-0.5, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -0.5], [-0.5, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -0.5], [-0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -0.5], [-1.0, 0.625, 0.5, 0.5, 0.5, 0.5, 0.625, -1.0], [-1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -1.0, -1.5]],
            4: [[-0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.25], [0.5, 1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.75, 0.25, 0.375, 0.75, 0.75, 0.375, 0.25, -0.5], [-1.0, -0.75, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0]],
            5: [[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]],
            6: [[-10.0, -9.0, -8.0, -8.0, -8.0, -8.0, -9.0, -10.0], [-7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0], [-5.0, -5.0, -6.0, -6.0, -6.0, -6.0, -5.0, -5.0], [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0], [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0], [0.5, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.5], [1.0, 2.0, 3.0, -1.0, -1.0, 0.0, 3.0, 1.0]]
        },
        stageLATE: {
            1: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.5, -1.5, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            2: [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, -0.875, 0.0, 1.0, 1.0, 0.0, -0.875, -1.5], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, -1.0], [-1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.5, -0.875, 0.0, 0.5, 0.5, 0.0, -0.875, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
            3: [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]],
            4: [[-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [1.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.25, 1.25, 1.0, 0.0, -0.5], [-0.5, 0.0, 0.75, 1.0, 1.0, 0.75, 0.0, -0.5], [-1.0, -0.5, 0.5, 0.75, 0.75, 0.5, -0.5, -1.0]],
            5: [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
            6: [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1.5], [-1.0, -0.5, 1.5, 1.5, 1.5, 1.5, -0.5, -1.0], [-1.0, -0.5, 1.5, 2.5, 2.5, 1.5, -0.5, -1.0], [-1.0, -0.5, 1.5, 2.5, 2.5, 1.5, -0.5, -1.0], [-1.0, -0.5, 1.5, 1.5, 1.5, 1.5, -0.5, -1.0], [-1.5, -1.25, -1.0, -0.75, -0.75, -1.0, -1.25, -1.5], [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]]
        }
    }
    HEATMAP[stageLATELATE] = HEATMAP[stageLATE] # LATELATE heatmap is the same as LATE heatmap

    MATERIAL: dict[int, float] = {
        chess.PAWN: 0.9,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.20,
        chess.QUEEN: 9.25,
        chess.KING: 0,
    }

    ESTIMATED_PAWN_VALUE = 40000

    MAX_QUIESCENCE_DEPTH = 4
    MAX_QUIESCENCE_DEPTH_LATE = 10
    MIN_PRUNE_SEARCH_DEPTH = 3 # SEARCH depth at which we start pruning the search tree
    # In other words, the tree is fully seearched x ply deep; higher values improve accuracy but seriously hurt performance

    def quiescence_extend_condition(self, board: chess.Board, move: chess.Move, stage: int) -> bool:
        if stage == Computer.stageLATELATE:
            return board.piece_at(move.from_square) == chess.PAWN
        elif stage == Computer.stageLATE:
            return board.is_capture(move) or board.gives_check(move) or board.piece_at(move.from_square) == chess.PAWN
        else:
            return board.is_capture(move) or board.gives_check(move)

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, *,
                original_depth: int = 0, current_path: list[chess.Move] | None = None,
                last_captured_square: chess.Square | None = None) -> tuple[float, list[chess.Move]]:
        """
        Evaluate the best move to make.

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
        :param current_path: The current path being explored (for optimal path tracking)
        :type current_path: list[chess.Move] | None
        :return: A tuple containing the best score and the optimal path (sequence of moves)
        :rtype: tuple[float, list[chess.Move]]
        """

        self.nodes_explored += 1
        search_depth = original_depth - depth
        enable_pruning = original_depth >= self.MIN_PRUNE_SEARCH_DEPTH

        # Initialize current_path if not provided
        if current_path is None:
            current_path = []

        # Check for timeout at the start of evaluation
        if self.is_timeup():
            return float('nan'), current_path  # Return current path to indicate timeout

        # Terminal node evaluation
        if board.is_checkmate():
            # If checkmate, return mate score adjusted for distance to prefer faster mates
            self.leaf_nodes_explored += 1
            # Use infinite score but adjust by depth to prefer faster mates
            if board.turn == chess.WHITE:
                # White to move and checkmated, so black delivered mate
                return float('-inf') + (original_depth - depth), current_path
            else:
                # Black to move and checkmated, so white delivered mate
                return float('inf') - (original_depth - depth), current_path
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_repetition(count=2):
            self.leaf_nodes_explored += 1
            return 0, current_path
        
        if self.can_syzygy(board):
            self.leaf_nodes_explored += 1
            return self.syzygy_score(board), current_path

        stage = self.get_game_stage(board.piece_map())

        # Quiescence search at leaf nodes
        if depth == 0:
            quiescence_score = self.quiescence_search(board, alpha, beta, 0, stage=stage)
            return quiescence_score, current_path
            # return self.evaluate(board), current_path

        is_maximizing = board.turn == chess.WHITE
        best_score = float('-inf') if is_maximizing else float('inf')
        best_path = current_path.copy()  # Start with current path

        # Extended Futility Pruning - more aggressive than standard futility
        if enable_pruning and depth <= 1 and not board.is_check():
            static_eval = self.evaluate(board)
            futility_margin = self.ESTIMATED_PAWN_VALUE * (depth * 2.0)  # More aggressive margin

            if is_maximizing and static_eval + futility_margin < alpha:
                self.alpha_cuts += 1
                self.prunes += 1
                return static_eval, current_path
            elif not is_maximizing and static_eval - futility_margin > beta:
                self.beta_cuts += 1
                self.prunes += 1
                return static_eval, current_path

        # Null Move Pruning - more conservative implementation
        # Disable for late game due to zugzwang
        if enable_pruning and depth >= 4 and not board.is_check() and stage != Computer.stageLATE:
            R = 2 #depth // 3  # Adjusted reduction for null move pruning
            board.push(chess.Move.null())
            null_score, null_path = self.minimax(board, depth - 1 - R, -beta, -beta + 1, 
                                    original_depth=original_depth, current_path=current_path)
            null_score = -null_score
            board.pop()
            
            if null_score >= beta:
                self.beta_cuts += 1
                self.prunes += 1
                return null_score, null_path

        # Move ordering
        legal_moves = list(board.legal_moves)
        
        # Get killer moves for current depth
        killer_moves = self.killer_moves.get(search_depth, [])
        
        # Separate moves: captures, killer moves, then others
        capture_moves = [move for move in legal_moves if board.is_capture(move)]
        killer_moves_in_legals = [move for move in killer_moves if move in legal_moves and move not in capture_moves]
        quiet_moves = [move for move in legal_moves if move not in capture_moves and move not in killer_moves_in_legals]
        
        # Order captures with MVV-LVA and SEE
        if capture_moves:
            capture_moves = self.see_ordering(board, capture_moves)
        
        # Order quiet moves with history heuristic
        if quiet_moves:
            quiet_moves = self.history_ordering(quiet_moves)
        
        # Combine moves: captures first, then killer moves, then quiet moves
        ordered_moves = capture_moves + killer_moves_in_legals + quiet_moves

        searched_moves = 0
        for move in ordered_moves:
            searched_moves += 1
            
            # Late Move Reduction (LMR)
            reduction = 0

            # Extension logic: extend search depth by 1 ply if conditions met
            extend = 0
            # Check if current board is in check (extend)
            if board.is_check():
                extend = 1
            # Recapture extension: if last captured square matches current move's to_square and move is capture
            elif last_captured_square is not None and board.is_capture(move) and move.to_square == last_captured_square:
                extend = 1
            # Pawn promotion extension
            elif move.promotion is not None:
                extend = 1

            if enable_pruning and depth >= 1 and searched_moves > 5 and not board.is_capture(move) and not board.gives_check(move):
                reduction = 1 + (searched_moves // 6)  # More reduction for later moves
                reduction = min(reduction, depth - 1)  # Don't reduce below depth 1

            # Adjust depth with extension
            new_depth = depth - 1 - reduction + extend

            board.push(move)
            
            # Create new path for this branch - ensure we include the current move
            new_path = current_path + [move]

            # Determine new last_captured_square for child node
            new_last_captured_square = None
            if board.is_capture(move):
                new_last_captured_square = move.to_square

            score, path = self.minimax(board, new_depth, alpha, beta, 
                                    original_depth=original_depth, current_path=new_path,
                                    last_captured_square=new_last_captured_square)
            board.pop()
            
            # Ensure the path returned from recursive call includes the current move
            # If path doesn't start with the current move, prepend it
            if path and path[0] != move:
                path = [move] + path
            
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_path = path  # Use the path returned from the recursive call
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_path = path  # Use the path returned from the recursive call
                beta = min(beta, best_score)

            # Alpha-beta pruning
            if beta <= alpha:
                # Store quiet moves as killer moves and update history
                if move not in capture_moves:
                    if search_depth not in self.killer_moves:
                        self.killer_moves[search_depth] = []
                    if move not in self.killer_moves[search_depth]:
                        self.killer_moves[search_depth].insert(0, move)
                        # Keep only 2 killer moves per depth
                        if len(self.killer_moves[search_depth]) > 2:
                            self.killer_moves[search_depth].pop()
                    
                    # Update history heuristic
                    self.history_table[move] = self.history_table.get(move, 0) + depth * depth

                self.beta_cuts += 1
                self.prunes += 1
                break

        return best_score, best_path
    
    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0, stage: int = 0) -> float:
        """
        Quiescence search with proper alpha-beta pruning.
        Only explores captures and checks.
        """

        self.nodes_explored += 1
        self.quiescence_nodes_explored += 1

        # Terminal conditions
        if board.is_checkmate():
            self.leaf_nodes_explored += 1
            return float('-inf') if board.turn == chess.WHITE else float('inf')
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_repetition(count=2):
            self.leaf_nodes_explored += 1
            return 0
        
        if self.can_syzygy(board):
            self.leaf_nodes_explored += 1
            return self.syzygy_score(board)

        stand_pat = self.evaluate(board)

        # Depth safeguard
        if (depth >= self.MAX_QUIESCENCE_DEPTH and stage < Computer.stageLATE) or depth >= self.MAX_QUIESCENCE_DEPTH_LATE:
            self.leaf_nodes_explored += 1
            return stand_pat

        is_maximizing = board.turn == chess.WHITE
        best_score = stand_pat

        # Alpha-beta pruning with stand_pat
        if is_maximizing:
            if best_score >= beta:
                self.beta_cuts += 1
                self.prunes += 1
                return best_score
            alpha = max(alpha, best_score)
        else:
            if best_score <= alpha:
                self.alpha_cuts += 1
                self.prunes += 1
                return best_score
            beta = min(beta, best_score)

        # Generate moves
        if board.is_check():
            moves = list(board.legal_moves)
        else:
            moves = [move for move in board.legal_moves if self.quiescence_extend_condition(board, move, stage)]

        # No moves? Return stand_pat
        if not moves:
            return best_score

        # Order moves
        moves = self.see_ordering(board, moves)#[:self.MAX_QUIESCENCE_MOVES]

        for move in moves:
            board.push(move)
            score = self.quiescence_search(board, alpha, beta, depth + 1, stage=stage)
            board.pop()

            if is_maximizing:
                if score > best_score:
                    best_score = score
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)

            # Cutoff check
            if alpha >= beta:
                self.prunes += 1
                if is_maximizing:
                    self.beta_cuts += 1
                else:
                    self.alpha_cuts += 1
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

        self.leaf_nodes_explored += 1

        transposition_key = board._transposition_key()
        if transposition_key in self.transposition_table:
            self.cache_hits += 1
            return self.transposition_table[transposition_key]

        def cse(x: float, y: float) -> float:
            """Complex safe exponentiation."""
            if x > 0:
                return x ** y
            else:
                return -(abs(x) ** y)

        # Cache piece map once
        piece_map = board.piece_map()

        stage = self.get_game_stage(piece_map)

        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king is None or black_king is None:
            return 0
        KINGS = {chess.WHITE: white_king, chess.BLACK: black_king}

        heatmap = self.HEATMAP[stage]

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

        piece_mobility_weight = {
            chess.PAWN: 0.1,
            chess.KNIGHT: 0.3,
            chess.BISHOP: 0.2,
            chess.ROOK: 0.15,
            chess.QUEEN: 0.12,
            chess.KING: 0.03
        }

        # Compute material and pieces
        PIECES: dict[bool, list[tuple[chess.Square, chess.Piece]]] = {chess.WHITE: [], chess.BLACK: []}
        material = {chess.WHITE: 1.0, chess.BLACK: 1.0}
        for sq, piece in piece_map.items():
            color = piece.color
            PIECES[color].append((sq, piece))
            material[color] += self.MATERIAL[piece.piece_type]
        num_pieces = len(piece_map)

        # Compute aggression
        aggression = {}
        for color in [chess.WHITE, chess.BLACK]:
            agg = min(material[color] / (2 * material[not color]), 1.5) ** 1.5
            agg *= 0.5 if stage == Computer.stageEARLY else 2 if stage == Computer.stageMIDDLE else 1.25
            aggression[color] = agg

        def piece_loop(color: chess.Color) -> dict:

            mobility_score = 0
            heatmap_score = 0
            minor_piece_development_bonus = 0
            aggression_score = 0
            attacks_cache: dict[chess.Square, int] = {}
            ranks = []

            # Precompute king position
            # Better to calculate distance ourselves since we already have rank and file
            krank = chess.square_rank(KINGS[not color])
            kfile = chess.square_file(KINGS[not color])

            for square, piece in PIECES[color]:
                ptype = piece.piece_type

                # Attack cache
                attacks_cache[square] = board.attacks_mask(square)

                # Mobility
                mobility = attacks_cache[square] & ~board.occupied_co[color]
                mobility_count = chess.popcount(mobility)
                weight = piece_mobility_weight[ptype]
                mobility_score += (math.log(mobility_count + 0.25, 1.65) - 1.5) * weight

                # Heatmap
                rank = square >> 3
                file = square & 7

                ranks.append(rank)

                heatmap_rank = 7 - rank if piece.color == chess.WHITE else rank
                if stage == Computer.stageLATE and ptype in {chess.KING, chess.PAWN}:
                    heatmap_score += heatmap[ptype][heatmap_rank][file] * 2
                else:
                    heatmap_score += heatmap[ptype][heatmap_rank][file]

                # MPDB
                if ptype in [chess.KNIGHT, chess.BISHOP]:
                    if square not in starting_squares[color][ptype]:
                        minor_piece_development_bonus += 1.5

                # Aggression
                dist = max(abs(rank - krank), abs(file - kfile))
                aggression_score += self.MATERIAL[ptype] * -(dist ** 1.5)

            # Get average rank and variance
            avg_rank = sum(ranks) / len(ranks)
            rank_var = sum((rank - avg_rank) ** 2 for rank in ranks) / len(ranks)
            rank_var = cse(rank_var, 0.5)

            return {
                "mobility_score" : mobility_score,
                "heatmap_score" : heatmap_score,
                "minor_piece_development_bonus" : minor_piece_development_bonus,
                "aggression_score" : aggression_score,
                "attacks_cache" : attacks_cache,
                "rank_var" : rank_var,
                "rank_avg" : avg_rank
            }

        def evaluate_player(color: chess.Color) -> float:
            king_square = KINGS[color]
            enemy_king_square = KINGS[not color]
            if king_square is None or enemy_king_square is None:
                return 0

            PIECE_SCORES = piece_loop(color)

            # Cache attacked squares and attacks per piece to avoid repeated calls
            attacks_cache = PIECE_SCORES["attacks_cache"]
            attacked_square_counts = collections.Counter()
            for attacks_mask in attacks_cache.values():
                attacked_square_counts.update(chess.scan_reversed(attacks_mask))

            # Legal moves are only important if it's the player's turn
            legal_captures = list(board.generate_legal_captures()) if color == board.turn else []

            def coverage() -> float:
                attack_bonus = 0

                center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
                
                # Calculate cover bonus denominator once
                cover_denominator = sum(self.MATERIAL[piece_map[sq].piece_type] for sq in attacks_cache.keys())
                cover_bonus = sum(attacked_square_counts.values()) / cover_denominator if cover_denominator > 0 else 0

                # Single loop through unique squares
                for square, count in attacked_square_counts.items():
                    attack_bonus += count ** 2

                    piece_type = board.piece_type_at(square)
                    rank = square >> 3

                    # Pre-calculate aggression multiplier
                    agg_mult = aggression[color]
                    
                    # Center control bonus
                    if square in center_squares:
                        attack_bonus += 2.5 * agg_mult
                    
                    # Enemy half bonus
                    if (color == chess.WHITE and rank > 3) or (color == chess.BLACK and rank < 4):
                        attack_bonus += 1.5 * agg_mult

                    if piece_type is None:
                        continue
                    
                    piece_color = board.color_at(square)

                    piece_weight_val = piece_mobility_weight[piece_type]
                    stage_mult = 7 if stage == Computer.stageLATE else 4
                    back_rank_mult = 6 if stage == Computer.stageLATE else 3
                    
                    # Enemy half piece bonus
                    if (color == chess.WHITE and rank > 3) or (color == chess.BLACK and rank < 4):
                        attack_bonus += piece_weight_val * agg_mult * stage_mult
                    
                    # Back rank penalty
                    if (color == chess.WHITE and rank < 2) or (color == chess.BLACK and rank > 5):
                        attack_bonus -= piece_weight_val * agg_mult * back_rank_mult
                    
                    # Attack/defense bonuses
                    if piece_color != color:
                        attack_bonus += self.MATERIAL[piece_type] ** 2 * agg_mult * 0.25
                    else:
                        attack_bonus += self.MATERIAL[piece_type] * 2 * aggression[not color]
                
                return attack_bonus + cover_bonus

            def control() -> float:
                control_bonus = 0
                for square, attacks_mask in attacks_cache.items():
                    control_bonus += chess.popcount(attacks_mask) ** 2 * 0.5

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
                def material_bonus_formula(m1: float, m2: float) -> float:
                    return min((m1 - m2) * (m1 / m2) ** 2 + m1, m1 * 3)

                base_score = material_bonus_formula(material[color], material[not color])
                return base_score + PIECE_SCORES["mobility_score"]

            def heatmap() -> float:
                return PIECE_SCORES["heatmap_score"]

            def minor_piece_bonus() -> float:
                return PIECE_SCORES["minor_piece_development_bonus"]

            def king_safety_penalty() -> float:

                def evaluate_pawn_shield() -> float:
                    """Evaluate the pawn shield in front of the king."""
                    shield_penalty = 0
                    king_rank = chess.square_rank(king_square)
                    king_file = chess.square_file(king_square)

                    # Define shield ranks based on color
                    if color == chess.WHITE:
                        shield_ranks = [king_rank + 1, king_rank + 2] if king_rank < 6 else [king_rank + 1]
                    else:
                        shield_ranks = [king_rank - 1, king_rank - 2] if king_rank > 1 else [king_rank - 1]

                    # Check files adjacent to king
                    shield_files = [king_file - 1, king_file, king_file + 1]

                    for rank in shield_ranks:
                        if rank < 0 or rank > 7:
                            continue
                        for file in shield_files:
                            if file < 0 or file > 7:
                                continue

                            square = chess.square(file, rank)
                            piece = board.piece_at(square)

                            if piece is None:
                                # Missing pawn in shield
                                shield_penalty += 15.0
                            elif piece.piece_type == chess.PAWN and piece.color == color:
                                # Friendly pawn in shield - good
                                shield_penalty -= 5.0
                            elif piece.piece_type == chess.PAWN and piece.color != color:
                                # Enemy pawn attacking shield
                                shield_penalty += 25.0

                    return shield_penalty

                def evaluate_open_files_near_king() -> float:
                    """Penalize open files near the king."""
                    open_file_penalty = 0
                    king_file = chess.square_file(king_square)

                    # Check files adjacent to king
                    check_files = [king_file - 1, king_file, king_file + 1]

                    for file in check_files:
                        if file < 0 or file > 7:
                            continue

                        file_bb = chess.BB_FILES[file]
                        pawns_on_file = board.pawns & file_bb

                        # If no pawns on file, it's open
                        if pawns_on_file == 0:
                            open_file_penalty += 20.0
                        else:
                            # Semi-open file (only enemy pawns)
                            enemy_pawns = pawns_on_file & board.occupied_co[not color]
                            friendly_pawns = pawns_on_file & board.occupied_co[color]

                            if friendly_pawns == 0 and enemy_pawns != 0:
                                open_file_penalty += 10.0

                    return open_file_penalty

                def evaluate_king_position() -> float:
                    """Penalize king position - center is bad, corners are good."""
                    position_penalty = 0
                    king_rank = chess.square_rank(king_square)
                    king_file = chess.square_file(king_square)

                    # Distance from center (files 3-4 are center)
                    file_distance = min(abs(king_file - 3), abs(king_file - 4))
                    rank_distance = min(abs(king_rank - 3), abs(king_rank - 4))

                    # Center penalty increases in middlegame
                    center_penalty = (file_distance + rank_distance) * 5.0
                    if stage == Computer.stageMIDDLE:
                        center_penalty *= 1.5

                    position_penalty += center_penalty

                    # Bonus for castled position
                    if has_moved and board.has_castling_rights(color):
                        position_penalty -= 10.0

                    return position_penalty

                def evaluate_enemy_tropism() -> float:
                    """Evaluate how close enemy pieces are to the king."""
                    tropism_penalty = 0

                    for square, piece in PIECES[not color]:
                        if piece.piece_type == chess.KING:
                            continue

                        # Calculate chebyshev distance to king
                        piece_rank = chess.square_rank(square)
                        piece_file = chess.square_file(square)
                        king_rank = chess.square_rank(king_square)
                        king_file = chess.square_file(king_square)

                        distance = max(abs(piece_file - king_file), abs(piece_rank - king_rank))

                        # Closer pieces are more threatening
                        if distance <= 2:
                            # Weight by piece value
                            threat_weight = self.MATERIAL[piece.piece_type] / 10.0
                            tropism_penalty += threat_weight * (3 - distance) ** 2

                    return tropism_penalty
                
                king_penalty = 0
                king_start_square = chess.E1 if color == chess.WHITE else chess.E8
                has_moved = king_square != king_start_square

                # Penalty for king having moved and lost castling rights
                if stage != Computer.stageLATE and (not board.has_castling_rights(color)) and has_moved:
                    king_penalty += 100.0

                # Bonus for king mobility (safe king moves)
                king_moves = list(chess.scan_reversed(attacks_cache.get(king_square, 0)))
                king_penalty -= len(king_moves) ** 0.5

                # Queen-like attacks on king zone
                attacks = chess.BB_DIAG_ATTACKS[king_square][chess.BB_DIAG_MASKS[king_square] & board.occupied]
                attacks |= (chess.BB_RANK_ATTACKS[king_square][chess.BB_RANK_MASKS[king_square] & board.occupied] |
                                chess.BB_FILE_ATTACKS[king_square][chess.BB_FILE_MASKS[king_square] & board.occupied])
                num_attacks = chess.popcount(attacks)
                king_penalty += num_attacks ** 2 * 0.5

                # Pawn shield evaluation
                # king_penalty += evaluate_pawn_shield()
                # Open files near king
                # king_penalty += evaluate_open_files_near_king()
                # King position penalty (center is bad)
                # king_penalty += evaluate_king_position()
                # Enemy piece tropism (pieces close to king)
                # king_penalty += evaluate_enemy_tropism()

                return king_penalty

            def pawn_structure() -> float:

                # Attempt to get from hash
                pawn_hash = pawn_hash = (
                    int(board.pawns & board.occupied_co[chess.WHITE]),
                    int(board.pawns & board.occupied_co[chess.BLACK])
                )

                if pawn_hash in self.pawn_hash_cache:
                    return self.pawn_hash_cache[pawn_hash]

                pawn_score = 0

                pawns_bb = board.pawns & board.occupied_co[color]
                enemy_pawns_bb = board.pawns & board.occupied_co[not color]

                pawn_files_bb = [chess.BB_FILES[file] & pawns_bb for file in range(8)]

                # Doubled pawns
                for file_bb in pawn_files_bb:
                    if chess.popcount(file_bb) > 1:
                        pawn_score -= 1.5

                # Isolated pawns
                files = [0x0101010101010101 << i for i in range(8)]
                isolated = 0
                for i, file_mask in enumerate(files):
                    if pawns_bb & file_mask:  # Pawns exist on this file
                        # Create mask for adjacent files
                        left_adj = files[i-1] if i > 0 else 0
                        right_adj = files[i+1] if i < 7 else 0
                        adjacent_files = left_adj | right_adj
                        
                        # If no pawns on adjacent files, count pawns on this file as isolated
                        if not (pawns_bb & adjacent_files):
                            isolated += chess.popcount(pawns_bb & file_mask)

                # Penalise for isolations
                pawn_score -= isolated * 3

                # Connected & passed pawns
                for square in chess.scan_reversed(pawns_bb):
                    rank = square >> 3
                    file = square & 7

                    # Create a bitmask of adjacent squares
                    adjacent_mask = 0
                    
                    # Add all possible adjacent squares (some may be off-board)
                    if square % 8 != 7:  # Not on h-file
                        adjacent_mask |= (1 << (square + 1))  # East
                        if square >= 8:  # Not on rank 1
                            adjacent_mask |= (1 << (square - 7))  # Northeast
                        if square < 56:  # Not on rank 8
                            adjacent_mask |= (1 << (square + 9))  # Southeast
                    
                    if square % 8 != 0:  # Not on a-file
                        adjacent_mask |= (1 << (square - 1))  # West
                        if square >= 8:  # Not on rank 1
                            adjacent_mask |= (1 << (square - 9))  # Northwest
                        if square < 56:  # Not on rank 8
                            adjacent_mask |= (1 << (square + 7))  # Southwest
                    
                    # Count how many pawns are on adjacent squares and reward
                    connections = chess.popcount(pawns_bb & adjacent_mask)
                    pawn_score += connections ** 1.25 * 1.5

                    num_protectors = chess.popcount(board.attackers_mask(color, square))

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
                            if (1 << square):
                                pawn_score += 1.0 * connections
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
                        distance_to_promotion = 7 - rank if color == chess.WHITE else rank
                        rank_bonus = (1 + (distance_to_promotion / 7)) ** 5
                        pieces_left_bonus = max(1 + 32 / num_pieces, 2.0)
                        # Score based on rank and number of protectors, add bonus for connections
                        pawn_score += (25.0 if stage == Computer.stageLATE else 4.0) * rank_bonus * num_protectors * pieces_left_bonus * (1 + connections) ** 2
                    
                    # Give a small exponential bonus for pawns closer to the end to encourage pushing closer pawns
                    if stage == Computer.stageLATELATE:
                        prom_rank = 7 if color == chess.WHITE else 0
                        distance = abs(prom_rank - rank)
                        pawn_score += distance ** -2 * num_protectors * (7.5 if stage == Computer.stageLATE else 1.5)

                # Store pawn hash in cache
                self.pawn_hash_cache[pawn_hash] = pawn_score
                return pawn_score

            def attack_quality() -> float:

                aggression_score = PIECE_SCORES["aggression_score"]
                if board.is_check():
                    if board.turn == color:
                        aggression_score -= 5
                    else:
                        aggression_score += 3

                # Penalise for high rank variance and low rank average
                rank_var = PIECE_SCORES["rank_var"]
                rank_avg = PIECE_SCORES["rank_avg"]

                dist_from_centre_rank = abs(rank_avg - 3.5)
                aggression_score -= rank_var * 2.0
                aggression_score -= dist_from_centre_rank * 2.0

                for move in legal_captures:
                    # Reward moves that attack opponent's pieces
                    victim = board.piece_type_at(move.to_square)
                    attacker = board.piece_type_at(move.from_square)
                    
                    # Skip if no victim or attacker (shouldn't happen since non-captures are skipped, 
                    # but it stops the errors showing up)
                    if not victim or not attacker:
                        continue

                    victim_material = self.MATERIAL[victim]
                    attacker_material = self.MATERIAL[attacker]

                    # Reward for attacking with pieces cheaper than the victim
                    if victim_material > attacker_material:
                        aggression_score += victim_material

                for square, piece in PIECES[not color]:
                    ptype = piece.piece_type
                    victim_value = self.MATERIAL[ptype]

                    # Get all attackers
                    attacker_mask = board.attackers_mask(color, square)
                    len_attackers = chess.popcount(attacker_mask)
                    if len_attackers == 0:
                        continue

                    cheapest_attacker_value = min(self.MATERIAL[board.piece_type_at(x) or 6] for x in chess.scan_reversed(attacker_mask))
                    # Reward for attacking with pieces cheaper than the victim
                    if victim_value > cheapest_attacker_value:
                        aggression_score += victim_value ** 2 * 0.5

                    defenders = board.attackers(not color, square)
                    if len(defenders) == 0:
                        # Reward for attackers on a piece with no defenders
                        aggression_score += victim_value * len_attackers * 2.0
                    else:
                        # Get cheapest defender
                        cheapest_defender_value = min(self.MATERIAL[board.piece_type_at(x) or 6] for x in defenders)
                        # Penalise for attacking with pieces more expensive than the defender and vice versa
                        if cheapest_attacker_value >= max(cheapest_defender_value, victim_value):
                            aggression_score -= cheapest_attacker_value ** 1.5
                        else:
                            aggression_score += victim_value ** 1.5

                    # Add bonus for pins and skewers (xray attacks)
                    for attacker_sq in chess.scan_reversed(attacker_mask):
                        attacker_ptype = board.piece_type_at(attacker_sq)
                        if not attacker_ptype or attacker_ptype in [chess.KNIGHT, chess.PAWN]:
                            continue  # Only sliding pieces for line attacks
                        # Calculate direction vector from attacker to victim
                        file_diff = chess.square_file(attacker_sq) - chess.square_file(square)
                        rank_diff = chess.square_rank(attacker_sq) - chess.square_rank(square)
                        if file_diff == 0:
                            direction = (0, 1 if rank_diff > 0 else -1)
                        elif rank_diff == 0:
                            direction = (1 if file_diff > 0 else -1, 0)
                        elif abs(file_diff) == abs(rank_diff):
                            direction = (1 if file_diff > 0 else -1, 1 if rank_diff > 0 else -1)
                        else:
                            continue  # Not a line attack
                        # Check if line between victim and attacker is clear
                        clear = True
                        cf, cr = chess.square_file(square) + direction[0], chess.square_rank(square) + direction[1]
                        while (cf, cr) != (chess.square_file(attacker_sq), chess.square_rank(attacker_sq)):
                            if board.piece_at(chess.square(cf, cr)):
                                clear = False
                                break
                            cf += direction[0]
                            cr += direction[1]
                        if not clear:
                            continue
                        # Check square behind victim in opposite direction
                        bf, br = chess.square_file(square) - direction[0], chess.square_rank(square) - direction[1]
                        if 0 <= bf < 8 and 0 <= br < 8:
                            behind_sq = chess.square(bf, br)
                            behind_piece = board.piece_at(behind_sq)
                            if behind_piece and behind_piece.color == (not color):
                                behind_value = self.MATERIAL[behind_piece.piece_type]
                                # Pin: behind piece more valuable than victim
                                # Reward slightly higher as pinned pieces are vulnerable
                                if behind_value > victim_value:
                                    bonus = behind_value ** 1.25 * 1.25
                                    aggression_score += bonus
                                # Skewer: victim more valuable than behind piece
                                elif victim_value > behind_value:
                                    bonus = behind_value ** 1.25
                                    aggression_score += bonus

                return aggression_score

            score = 0
            if stage != Computer.stageLATE:
                score -= cse(king_safety_penalty(), 1.35) * 5.5
            score += cse(material_score(), 2.5) * (85 if stage == Computer.stageLATE else 70)
            score += cse(coverage(), 1.75) * aggression[color] * (6 if stage == Computer.stageLATE else 4)
            score += cse(heatmap(), (1.75 if stage != Computer.stageMIDDLE else 1.5)) * aggression[not color] * (60 if stage == Computer.stageLATE else 50 if stage == Computer.stageEARLY else 40)
            score += cse(control(), 1.35) * aggression[color] * (15 if stage == Computer.stageLATE else 10 if stage == Computer.stageEARLY else 7)
            score += minor_piece_bonus() * 20 * aggression[color]
            score += cse(pawn_structure(), 2.25) * (15 if stage == Computer.stageLATE else 10)
            score += cse(attack_quality(), 1.5) * aggression[color] * (1 if stage == Computer.stageLATE else 6)

            if isinstance(score, complex):
                print("\nAGG", aggression[color])
                print("EAG", aggression[not color])
                print("-KSP", cse(king_safety_penalty(), 1.75) * (0.75 if stage == Computer.stageLATE else 2))
                print("+MS", cse(material_score(), 2.5) * material_score() * (75 if stage == Computer.stageLATE else 50))
                print("+COV", cse(coverage(), 1.5) * aggression[color] * (5.5 if stage == Computer.stageLATE else 2))
                print("+HEAT", cse(heatmap(), (1.5 if stage != Computer.stageMIDDLE else 1.25)) * aggression[not color] * (35 if stage == Computer.stageLATE else 25 if stage == Computer.stageEARLY else 20))
                print("+CTRL", cse(control(), 1.25) * aggression[color] * (12 if stage == Computer.stageLATE else 6.5 if stage == Computer.stageEARLY else 4.5))
                print("+MIN", minor_piece_bonus() * 15 * aggression[color])
                print("+PAWN", cse(pawn_structure(), 2) * (10 if stage == Computer.stageLATE else 6.5))
                print("+ATT", attack_quality() * aggression[color] * (3.75 if stage == Computer.stageLATE else 15))
                raise ValueError("Score is complex")

            return score

        score = 0
        score += evaluate_player(chess.WHITE)
        score -= evaluate_player(chess.BLACK)

        self.transposition_table[transposition_key] = score

        return score

    def evaluate_move_parallel(self, args):
        """
        Helper function for parallel move evaluation.
        This function must be defined at module level to be picklable for multiprocessing.
        """
        move_uci, board_fen, depth_val, reduction_val, start_time, timeout, shared_transposition_table, shared_pawn_hash_cache = args
        move = chess.Move.from_uci(move_uci)
        board = chess.Board(board_fen)
        
        # Create a temporary computer instance for evaluation with shared caches
        computer = Computer(board.turn, shared_transposition_table=shared_transposition_table, shared_pawn_hash_cache=shared_pawn_hash_cache)
        computer.start_time = start_time
        computer.timeout = timeout
        
        # Check timeout before starting evaluation
        if computer.is_timeup():
            return (move_uci, float('nan'), {
                'nodes_explored': 0,
                'leaf_nodes_explored': 0,
                'alpha_cuts': 0,
                'beta_cuts': 0,
                'prunes': 0,
                'cache_hits': 0,
                'optimal_path': []
            })
        
        board.push(move)
        score, optimal_path = computer.minimax(board, depth_val - reduction_val, float('-inf'), float('inf'), original_depth=depth_val - reduction_val, current_path=[])
        board.pop()
        
        # Return both the score and the metrics from this evaluation
        return (move_uci, score, {
            'nodes_explored': computer.nodes_explored,
            'leaf_nodes_explored': computer.leaf_nodes_explored,
            'alpha_cuts': computer.alpha_cuts,
            'beta_cuts': computer.beta_cuts,
            'prunes': computer.prunes,
            'cache_hits': computer.cache_hits,
            'optimal_path': optimal_path
        })

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
            return opening_best
        
        move = self.best_syzygy(board)
        if move is not None:
            print("Using Syzygy:",board.san(move))
            return move
    
        return None

    def calculate_threshold(self, num_moves: int, board: chess.Board, depth: int) -> float:
        """
        Calculate the threshold value for the number of moves.

        :param num_moves: The number of moves in the game
        :type num_moves: int
        :param board: The current state of the board
        :type board: chess.Board
        :param depth: The current depth of the search tree
        :type depth: int
        :return: The calculated threshold value
        :rtype: float
        """

        if self.get_game_stage(board.piece_map()) != Computer.stageLATE:
            threshold = 0.1
        else:
            threshold = 0.2
        threshold *= ((len(list(board.legal_moves)) / num_moves)) ** 0.5
        threshold = min(threshold, 0.85)

        return threshold
        
    def best_move(self, board: chess.Board, timeout: float=float('inf'), time_per_move: float | None = None) -> chess.Move | None:
        
        """
        Determine and return the best move for the computer player using the Minimax algorithm.

        :param board: The current state of the board
        :type board: chess.Board
        :param timeout: The maximum allowed time to find the best move
        :type timeout: float
        :param time_per_move: The allowed time per move. If specified, overrides the timeout
        :type time_per_move: float
        :return: The best legal move for the computer player, or None if no move is possible
        :rtype: chess.Move | None
        """

        def _should_terminate(move_score_map: list[tuple[chess.Move, float]]) -> bool:
            return any(score == self.BEST_SCORE for _, score in move_score_map)

        instant_response = self.instant_response(board)
        if instant_response is not None:
            return instant_response
        
        ####################################################################################################


        self.start_time = time.time()
        self.timeout = time_per_move or self.allocated_time(timeout, board.fullmove_number)
        stage = self.get_game_stage(board.piece_map())
        formatted_stage = {self.stageEARLY: "opening", self.stageMIDDLE: "midgame", self.stageLATE: "endgame", self.stageLATELATE: "late endgame"}[stage]

        print(f"""
        - It's {"white" if board.turn == chess.WHITE else "black"}'s move
        - I will think for {self.timeout:.2f}s
        - I think the game is in the {formatted_stage}
        """)

        depth = 0
        best_move = None
        reset = False

        # Reset metrics
        self.reset_metrics()
        
        maxmin = max if board.turn == chess.WHITE else min

        moves = list(board.legal_moves)  # Convert generator to list for membership checks
        best_move = chess.Move.null()

        move_score_map: list[tuple[chess.Move, float]] = [(move, 0) for move in moves] # Initialize once before loop
        optimal_path: list[chess.Move] = []

        # Sort by the MVV-LVA heuristic
        move_score_map.sort(key=lambda x: self.mvv_lva_score(board, x[0]), reverse=True)

        while not self.is_timeup():

            print(f"""\nDEPTH {depth}: """,end='\t')

            move_score_map.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
            moves = [move for move, _ in move_score_map]

            # Gradually filter out based on the previous scores
            if not reset and depth > 1:
                threshold = self.calculate_threshold(len(moves), board, depth)

                # If best move is a capture, increase the threshold since it is likely the best choice
                if board.is_capture(best_move):
                    threshold *= 1.25
                
                print("Threshold:",threshold)
                turning_point = self._turning_point([score for _, score in move_score_map], threshold=threshold, depth=depth)
                move_score_map = move_score_map[:turning_point]
                moves = moves[:turning_point]
            print(len(moves),"moves:",[board.san(m) for m in moves])

            # If only one move left, return it
            if len(moves) == 1:
                formatted_path = self.display_optimal_path(board, [moves[0]] + optimal_path)
                self.display_metrics(formatted_path)
                self.print_board_feedback(board, move_score_map)
                return moves[0]

            # Create a dictionary for quick lookup and update of scores
            move_score_dict = dict(move_score_map)

            current_best_move = None
            current_best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

            # Use multiprocessing for parallel move evaluation
            # Only use parallel search for deeper searches with multiple moves and if time is not low
            if self.workers > 1 and len(moves) > 5 and depth >= 5 and time.time() - self.start_time < (self.timeout * 0.75):
            
                # Create shared caches for parallel evaluation
                manager = mp.Manager()
                shared_transposition_table = manager.dict()
                shared_pawn_hash_cache = manager.dict()
                
                # Prepare arguments for parallel evaluation
                args_list = []
                for i, move in enumerate(moves):
                    # Apply LMR reduction
                    reduction = 0
                    # if depth > 3 and stage != Computer.stageLATE:
                    #     lmr_context = {'search_depth': depth, 'move_count': len(move_score_map), 'game_stage': stage}
                    #     if i > self._turning_point([score for _, score in move_score_map], threshold=1.5 - (i / len(moves)), context=lmr_context):
                    #         reduction += 1
                    
                    args_list.append((move.uci(), board.fen(), depth, reduction, self.start_time, self.timeout, 
                                    shared_transposition_table, shared_pawn_hash_cache))
                
                # Create process pool
                print("MP with",self.workers,"processes")
                with mp.Pool(processes=min(self.workers, len(moves))) as pool:
                    # Evaluate moves in parallel using the helper function
                    results = []
                    try:
                        # Use map_async to allow early termination
                        async_result = pool.map_async(self.evaluate_move_parallel, args_list, chunksize=1)
                        
                        # Wait for results with timeout, checking periodically for timeout
                        timeout_remaining = max(0, self.timeout - (time.time() - self.start_time)) if self.timeout != float('inf') else None
                        while not async_result.ready():
                            if self.is_timeup():
                                print("TIMEOUT during parallel evaluation - terminating pool")
                                pool.terminate()
                                break
                            time.sleep(0.01)  # Small sleep to avoid busy waiting
                        
                        if not self.is_timeup():
                            results = async_result.get(timeout=0.1)
                    except Exception as e:
                        print(f"Parallel evaluation error: {e}")
                        # Fall back to sequential evaluation
                        results = []
                    
                # Update main cache with results from shared cache
                self.transposition_table.update(shared_transposition_table)
                self.pawn_hash_cache.update(shared_pawn_hash_cache)
            
                # Process results
                for result in results:
                    move_uci, score, metrics = result
                    if self.is_timeup():
                        print("TIMEUP")
                        break
                    
                    move = chess.Move.from_uci(move_uci)

                    # Normalise scores as they grow extremely large
                    print(f"{board.san(move)} : {self.normalise_score(score)}",end='\t',flush=True)
                    
                    move_score_dict[move] = score

                    # Aggregate metrics from parallel evaluation
                    self.nodes_explored += metrics['nodes_explored']
                    self.leaf_nodes_explored += metrics['leaf_nodes_explored']
                    self.alpha_cuts += metrics['alpha_cuts']
                    self.beta_cuts += metrics['beta_cuts']
                    self.prunes += metrics['prunes']
                    self.cache_hits += metrics['cache_hits']
                    optimal_path: list[chess.Move] = metrics['optimal_path']

                    # Update current best if needed
                    if board.turn == chess.WHITE:
                        if score > current_best_score:
                            current_best_score = score
                            current_best_move = move
                    else:
                        if score < current_best_score:
                            current_best_score = score
                            current_best_move = move

                    # Remove moves that lead to checkmate
                    if score == self.WORST_SCORE:
                        del move_score_dict[move]

                    if _should_terminate(list(move_score_dict.items())):
                        print("TERMINATED EARLY")
                        break
            else:
                # Sequential evaluation for shallow searches or single moves
                for move in moves:

                    board.push(move)
                    score, optimal_path = self.minimax(board, depth, float('-inf'), float('inf'), original_depth=depth, current_path=[])
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
                return best_move

            # Update move_score_map from the dictionary for next iteration
            move_score_map = list(move_score_dict.items())

            # Terminate early if an immediate win is found
            if _should_terminate(move_score_map):

                # Choose best move from current depth
                best_move = current_best_move if current_best_move is not None else maxmin(move_score_map, key=lambda x: x[1])[0]
                break

            best_move = current_best_move if current_best_move is not None else maxmin(move_score_map, key=lambda x: x[1])[0]

            print("BEST:",board.san(best_move))

            depth += 1

        formatted_path = self.display_optimal_path(board, [best_move or chess.Move.null()] + optimal_path)
        self.display_metrics(formatted_path)
        self.print_board_feedback(board, move_score_map)

        return best_move

    # def ponder(self, board: chess.Board)

    ##################################################
    #                     EXTRAS                     #
    ##################################################

    TIME_IF_INF = 60

    def get_game_stage(self, piece_map: dict) -> int:
        """Return the current stage of the game."""

        num_pieces = len([piece for piece in piece_map.values() if piece.piece_type != chess.PAWN])
        if num_pieces >= 12:
            return Computer.stageEARLY
        elif num_pieces >= 8:
            return Computer.stageMIDDLE
        elif num_pieces > 2:
            return Computer.stageLATE
        else:
            return Computer.stageLATELATE

    def allocated_time(self, t: float, m: int) -> float:
        """
        Determine the time the computer should use based on the time remaining and the current move number.

        :param t: The total time remaining in seconds
        :type t: float
        :param m: The number of full moves
        :type m: int
        :return: The allocated time in seconds
        :rtype: float
        """
        if t == float('inf'):
            return self.TIME_IF_INF
        upper_limit = t * 0.075
        return min(upper_limit, max(0.05, (t / (20 + (40 - m)/2) + 5)))

    def is_timeup(self) -> bool:
        if self.timeout is None or self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        # Add a small buffer to ensure we don't exceed the timeout
        return elapsed > (self.timeout - 0.1)

    def normalise_score(self, score: float) -> float:
        return round(score / self.ESTIMATED_PAWN_VALUE, 2)

    def reset_metrics(self) -> None:
        self.nodes_explored = 0
        self.leaf_nodes_explored = 0
        self.quiescence_nodes_explored = 0
        self.alpha_cuts = 0
        self.beta_cuts = 0
        self.prunes = 0
        self.cache_hits = 0

    def display_metrics(self, optimal_path: list[str]) -> None:
        
        elapsed = time.time() - self.start_time if self.start_time is not None else float("nan")
        cache_hit_proportion = self.cache_hits / self.leaf_nodes_explored if self.leaf_nodes_explored > 0 else float("nan")
        print(f"""
        Nodes explored      : {self.nodes_explored} | NPS : {self.nodes_explored / elapsed:.2f}
        Qsc nodes explored  : {self.quiescence_nodes_explored} | QNPS : {self.quiescence_nodes_explored / elapsed:.2f}
        Leaf nodes explored : {self.leaf_nodes_explored} | LNPS : {self.leaf_nodes_explored / elapsed:.2f}
        Prunes:
            Alpha : {self.alpha_cuts}
            Beta  : {self.beta_cuts}
            Total : {self.prunes}
        Cache hits : {self.cache_hits} | {cache_hit_proportion * 100:.2f}%
        Time elapsed: {elapsed:.2f}s
        
        Optimal path: {optimal_path}
        
        """)

    @staticmethod
    def estimate_pawn_value(depth: int) -> None:
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

            normal_score, _ = c.minimax(chess.Board(normal_fen), i + 1, float('-inf'), float('inf'), original_depth=i + 1)
            no_pawn_score, _ = c.minimax(chess.Board(fen_without_pawn), i + 1, float('-inf'), float('inf'), original_depth=i + 1)

            c.display_metrics([])
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
        score = self.normalise_score(scoremap[0][1])
        score = score if board.turn == chess.WHITE else -score

        print(f"""
{__version__.capitalize()}: I evaluate the position at {score} in my favour.
""")

    @staticmethod
    def cutechess() -> None:
        """UCI (Universal Chess Interface) protocol implementation."""
        board = chess.Board()
        computer = None
        search_stop_requested = False
        current_search_thread = None
        
        # Engine options
        options = {
            "Hash": 128,  # MB
            "Threads": 1,
            "MultiPV": 1,
            "SyzygyPath": "",
            "SyzygyProbeDepth": 1,
            "Syzygy50MoveRule": True,
            "UseNNUE": False,
            "Contempt": 0,
        }
        
        def create_computer():
            """Create a new computer instance with the correct color."""
            nonlocal computer
            computer = Computer(board.turn)
            
        create_computer()
        
        def parse_go_parameters(tokens: list[str]) -> dict:
            """Parse go command parameters and return a dictionary of search settings."""
            params = {
                "movetime": None,
                "wtime": None,
                "btime": None,
                "winc": None,
                "binc": None,
                "depth": None,
                "nodes": None,
                "infinite": False
            }
            
            i = 1
            while i < len(tokens):
                token = tokens[i]
                if token == "movetime" and i + 1 < len(tokens):
                    params["movetime"] = int(tokens[i + 1]) / 1000.0  # Convert ms to seconds
                    i += 2
                elif token == "wtime" and i + 1 < len(tokens):
                    params["wtime"] = int(tokens[i + 1]) / 1000.0
                    i += 2
                elif token == "btime" and i + 1 < len(tokens):
                    params["btime"] = int(tokens[i + 1]) / 1000.0
                    i += 2
                elif token == "winc" and i + 1 < len(tokens):
                    params["winc"] = int(tokens[i + 1]) / 1000.0
                    i += 2
                elif token == "binc" and i + 1 < len(tokens):
                    params["binc"] = int(tokens[i + 1]) / 1000.0
                    i += 2
                elif token == "depth" and i + 1 < len(tokens):
                    params["depth"] = int(tokens[i + 1])
                    i += 2
                elif token == "nodes" and i + 1 < len(tokens):
                    params["nodes"] = int(tokens[i + 1])
                    i += 2
                elif token == "infinite":
                    params["infinite"] = True
                    i += 1
                else:
                    i += 1
            
            return params
        
        def calculate_timeout(params: dict) -> float:
            """Calculate appropriate timeout based on time control parameters."""
            if params["movetime"] is not None:
                return params["movetime"]
                
            if params["infinite"]:
                return float('inf')
                
            # Time control calculation
            time_remaining = params["wtime"] if board.turn == chess.WHITE else params["btime"]
            time_increment = params["winc"] if board.turn == chess.WHITE else params["binc"]
            
            if time_remaining is not None:
                # Simple time management: use 1/40th of remaining time + increment
                moves_to_go = 40  # Estimate 40 moves remaining
                base_time = time_remaining / moves_to_go
                if time_increment is not None:
                    base_time += time_increment
                # Don't use more than 1/4 of remaining time
                return min(base_time, time_remaining * 0.25)
                
            return 10.0  # Default timeout if no time control specified
        
        def reset_engine():
            """Reset the engine state for a new game."""
            nonlocal computer, search_stop_requested
            computer = Computer(board.turn)
            search_stop_requested = False
            # Clear caches for new game
            computer.transposition_table.clear()
            computer.pawn_hash_cache.clear()
            computer.history_table.clear()
            computer.killer_moves.clear()
        
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                print("info string Engine stopped by user")
                break

            # Handle UCI commands
            if line == "uci":
                print(f"id name {NAME} {__version__}")
                print("id author Vai")
                # Print available options
                print("option name Hash type spin default 128 min 1 max 1024")
                print("option name Threads type spin default 1 min 1 max 1")  # Single-threaded for now
                print("option name MultiPV type spin default 1 min 1 max 1")   # Single PV for now
                print("option name SyzygyPath type string default \"\"")
                print("option name SyzygyProbeDepth type spin default 1 min 1 max 100")
                print("option name Syzygy50MoveRule type check default true")
                print("option name UseNNUE type check default false")
                print("option name Contempt type spin default 0 min -100 max 100")
                print("uciok")

            elif line == "isready":
                print("readyok")

            elif line == "ucinewgame":
                reset_engine()
                print("info string New game started")

            elif line.startswith("setoption"):
                tokens = line.split()
                if len(tokens) >= 5 and tokens[1] == "name" and tokens[3] == "value":
                    option_name = tokens[2]
                    option_value = tokens[4]
                    
                    if option_name in options:
                        try:
                            if isinstance(options[option_name], bool):
                                options[option_name] = option_value.lower() == "true"
                            elif isinstance(options[option_name], int):
                                options[option_name] = int(option_value)
                            elif isinstance(options[option_name], str):
                                options[option_name] = option_value
                            print(f"info string Set option {option_name} = {option_value}")
                        except ValueError:
                            print(f"info string Invalid value for option {option_name}: {option_value}")
                    else:
                        print(f"info string Unknown option: {option_name}")
                else:
                    print("info string Invalid setoption command format")

            elif line.startswith("position"):
                tokens = line.split()
                if "startpos" in tokens:
                    board = chess.Board()
                    moves_index = tokens.index("startpos") + 1
                elif "fen" in tokens:
                    fen_index = tokens.index("fen") + 1
                    # FEN can have 6 parts, but we need to handle variable length
                    fen_parts = []
                    i = fen_index
                    while i < len(tokens) and tokens[i] != "moves":
                        fen_parts.append(tokens[i])
                        i += 1
                    fen_str = " ".join(fen_parts)
                    try:
                        board = chess.Board(fen=fen_str)
                        moves_index = i
                    except ValueError:
                        print("info string Invalid FEN")
                        continue
                else:
                    moves_index = len(tokens)

                if "moves" in tokens:
                    moves_idx = tokens.index("moves") + 1
                    for move_str in tokens[moves_idx:]:
                        try:
                            move = chess.Move.from_uci(move_str)
                            if move in board.legal_moves:
                                board.push(move)
                            else:
                                print(f"info string Illegal move: {move_str}")
                                break
                        except ValueError:
                            print(f"info string Invalid move: {move_str}")
                            break
                
                # Update computer with current position's turn
                create_computer()

            elif line.startswith("go"):
                search_stop_requested = False
                tokens = line.split()
                params = parse_go_parameters(tokens)
                
                # Calculate timeout based on parameters
                timeout = calculate_timeout(params)
                
                # For fixed depth search, we'll use a simplified approach
                # since the current best_move implementation doesn't support fixed depth directly
                if params["depth"] is not None:
                    # Create a temporary computer instance for fixed depth search
                    temp_computer = Computer(board.turn)
                    # Set a very high timeout since we want to complete the fixed depth
                    move = temp_computer.best_move(board, timeout=float('inf'))
                else:
                    # Normal time-controlled search - ensure computer is properly initialized
                    create_computer()  # Ensure computer is initialized
                    move = computer.best_move(board, timeout=timeout) # type: ignore
                
                if move is None:
                    # If no legal moves, check if game is over
                    if board.is_game_over():
                        if board.is_checkmate():
                            print("info string Checkmate")
                        elif board.is_stalemate():
                            print("info string Stalemate")
                        else:
                            print("info string Game over")
                        print("bestmove resign")
                    else:
                        print("bestmove 0000")  # Null move (shouldn't happen)
                else:
                    print(f"bestmove {move.uci()}")

            elif line == "stop":
                search_stop_requested = True
                print("info string Search stopped")

            elif line == "quit":
                sys.exit(0)

            else:
                print(f"info string Unknown command: {line}")

    @staticmethod
    def profile_evaluation() -> None:

        import cProfile

        c = Computer(chess.WHITE)

        def generate_positions(n: int=100, moves: int=100):
            for i in range(n):
                board = chess.Board()
                for _ in range(moves):
                    if not board.is_game_over():
                        board.push(rnd.choice(list(board.legal_moves)))
                print(i, end='\r', flush=True)
                yield board

        print("Generating...")
        boards = list(generate_positions(n=1000, moves=100))
        print("Generated.")

        print("Warming up the interpreter...")
        for _ in range(int(3e4)):
            _ = 12345 ** 2345
        print("Interpreter warmed up.")

        def evaluate_boards() -> None:
            for board in boards:
                c.transposition_table.clear()
                c.evaluate(board)

        cProfile.runctx('evaluate_boards()', globals(), locals(), sort='cumtime')

    def display_optimal_path(self, board: chess.Board, optimal_path: list[chess.Move]) -> list[str]:
        path = []
        board_copy = board.copy()
        for move in optimal_path:
            try:
                path.append(board_copy.san(move))
                board_copy.push(move)
            except AssertionError:
                path.append(move.uci())
        return path


def main():

    FEN = chess.STARTING_FEN
    # FEN = "8/6kp/2p5/1p2p3/8/1P4PP/P4K2/8 w - - 0 37"

    board = chess.Board(FEN)
    players = [Computer(board.turn, 7), Computer(not board.turn, 1)]

    while not board.is_game_over():
        print(board,"\n\n")
        player = players[0] if board.turn == chess.WHITE else players[1]
        move = player.best_move(board, time_per_move=20)
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
    # Computer.profile_evaluation()
    # Computer.estimate_pawn_value(6)
