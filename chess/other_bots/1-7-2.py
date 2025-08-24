"""
This bot is similar to 1.7.1b-simple, but using 1.7's evaluation function with major optimisations.

Errors in the pruning during minimax have also been fixed, reducing blunders.
"""

import chess
import collections
import random as rnd
import requests
import time
import urllib.parse

from sys import setrecursionlimit
from typing import Hashable

setrecursionlimit(int(1e6))

__version__ = '1.7.2'

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

        # Metrics
        self.nodes_explored = 0
        self.leaf_nodes_explored = 0
        self.alpha_cuts = 0
        self.beta_cuts = 0
        self.prunes = 0
        self.cache_hits = 0

        # Evaluation caches
        self.transposition_table: dict[Hashable, float] = {}
        self.pawn_hash_cache: dict[tuple[int, int], float] = {}

    ##################################################
    #            OPENING BOOKS AND syzygy            #
    ##################################################

    syzygy_URL = "https://tablebase.lichess.ovh/standard?fen="
    OPENING_URL = "https://explorer.lichess.ovh/master?fen="

    OPENING_LEAVE_CHANCE = 0.05  # Chance to leave the opening book

    def can_syzygy(self, board: chess.Board, best_score: float) -> bool:
        num_pieces = bin(board.occupied).count('1')
        if num_pieces > 7:
            return False

        win_threshold = 100
        lose_threshold = -50
        if self.normalise_score(best_score) >= win_threshold or self.normalise_score(best_score) <= lose_threshold:
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

        url = self.syzygy_URL + fen_encoded

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.RequestException(f"Request to Syzygy tablebase server failed with status code {response.status_code}")

    def best_syzygy(self, board: chess.Board, best_score: float) -> chess.Move | None:
        """
        Get the best move from the Syzygy tablebase server for the given board position.

        Args:
            board (chess.Board): The chess board position to get the best move for.
            best_score (float): The best score for the board position.

        Returns:
            chess.Move: The best move from the Syzygy tablebase server.
            None: If no best move is found.
        """

        if not self.can_syzygy(board, best_score):
            return None

        response = self.syzygy_query(board)
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
            return 1

        score_range = max(scores) - min(scores)
        if score_range == 0:
            return 1

        for i in range(len(scores) - 1):
            gap = abs(scores[i] - scores[i + 1])
            # Cut the gap if it is at least 50 points wide
            if gap > 50 * self.ESTIMATED_PAWN_VALUE:
                return i + 1
            # Check if the gap is greater than the threshold and greater than 100 centipawns
            if gap / score_range >= threshold and gap > self.ESTIMATED_PAWN_VALUE:
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

    stageEARLY = 1
    stageMIDDLE = 2
    stageLATE = 3

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

    MATERIAL: dict[int, float] = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.00,
        chess.QUEEN: 9.25,
        chess.KING: 0
    }

    ESTIMATED_PAWN_VALUE = 10000

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

        # Terminal node evaluation
        if board.is_checkmate():
            # If checkmate, return mate score adjusted for distance
            self.leaf_nodes_explored += 1
            return float('-inf') if board.turn == chess.WHITE else float('inf')
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.is_repetition(count=3):
            self.leaf_nodes_explored += 1
            return 0

        # Leaf node evaluation
        if depth == 0:
            self.leaf_nodes_explored += 1
            return self.evaluate(board)

        is_maximizing = board.turn == chess.WHITE
        best_score = float('-inf') if is_maximizing else float('inf')

        # Futility pruning - only at frontier nodes and when not in check
        if depth <= 2 and not board.is_check():
            static_eval = self.evaluate(board)
            margin = self.ESTIMATED_PAWN_VALUE * 1.5  # More conservative margin
            
            if is_maximizing and static_eval + margin < alpha:
                self.alpha_cuts += 1
                self.prunes += 1
                return static_eval + margin
            elif not is_maximizing and static_eval - margin > beta:
                self.beta_cuts += 1
                self.prunes += 1
                return static_eval - margin

        # Null Move Pruning - more conservative implementation
        if depth >= 3 and not board.is_check() and not any(board.is_capture(move) for move in board.legal_moves):
            R = 2  # More conservative reduction
            board.push(chess.Move.null())
            null_score = -self.minimax(board, depth - 1 - R, -beta, -alpha, 
                                    original_depth=original_depth, 
                                    heuristic_sort=heuristic_sort, 
                                    heuristic_eliminate=heuristic_eliminate, 
                                    use_mvv_lva=use_mvv_lva)
            board.pop()
            
            if null_score >= beta:
                self.beta_cuts += 1
                self.prunes += 1
                return null_score

        # Move ordering
        legal_moves = list(board.legal_moves)
        
        # Get killer moves for current depth
        search_depth = original_depth - depth
        killer_moves = self.killer_moves.get(search_depth, [])
        
        # Separate moves: captures, killer moves, then others
        capture_moves = [move for move in legal_moves if board.is_capture(move)]
        killer_moves_in_legals = [move for move in killer_moves if move in legal_moves and move not in capture_moves]
        quiet_moves = [move for move in legal_moves if move not in capture_moves and move not in killer_moves_in_legals]
        
        # Order captures with MVV-LVA if enabled
        if use_mvv_lva and capture_moves:
            capture_moves = self.mvv_lva_ordering(board, capture_moves)
        
        # Combine moves: captures first, then killer moves, then quiet moves
        ordered_moves = capture_moves + killer_moves_in_legals + quiet_moves

        for move in ordered_moves:
            board.push(move)
            score = self.minimax(board, depth - 1, alpha, beta, 
                                original_depth=original_depth,
                                heuristic_sort=heuristic_sort,
                                heuristic_eliminate=heuristic_eliminate,
                                use_mvv_lva=use_mvv_lva)
            board.pop()
            
            if is_maximizing:
                if score > best_score:
                    best_score = score
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)

            # Alpha-beta pruning
            if beta <= alpha:
                # Store quiet moves as killer moves
                if not board.is_capture(move) and not board.gives_check(move):
                    if search_depth not in self.killer_moves:
                        self.killer_moves[search_depth] = []
                    if move not in self.killer_moves[search_depth]:
                        self.killer_moves[search_depth].insert(0, move)
                        # Keep only 2 killer moves per depth
                        if len(self.killer_moves[search_depth]) > 2:
                            self.killer_moves[search_depth].pop()

                self.beta_cuts += 1
                self.prunes += 1
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

        piece_weight = {
            chess.PAWN: 0.5,
            chess.KNIGHT: 1.5,
            chess.BISHOP: 1.7,
            chess.ROOK: 1.2,
            chess.QUEEN: 1.0,
            chess.KING: 0.3
        }

        # Compute material and pieces
        PIECES: dict[bool, list[tuple[chess.Square, chess.Piece]]] = {chess.WHITE: [], chess.BLACK: []}
        material = {chess.WHITE: 1.0, chess.BLACK: 1.0}
        for sq, piece in piece_map.items():
            color = piece.color
            PIECES[color].append((sq, piece))
            material[color] += self.MATERIAL[piece.piece_type]

        # Compute aggression
        aggression = {}
        for color in [chess.WHITE, chess.BLACK]:
            agg = min(material[color] / (2 * material[not color]), 1.5) ** 2
            agg *= 0.5 if stage == Computer.stageEARLY else 1.75 if stage == Computer.stageMIDDLE else 1.25
            aggression[color] = agg

        def piece_loop(color: chess.Color) -> dict:

            mobility_score = 0
            heatmap_score = 0
            minor_piece_development_bonus = 0
            aggression_score = 0
            attacks_cache: dict[chess.Square, chess.SquareSet] = {}

            # Precompute king position
            # Better to calculate distance ourselves since we already have rank and file
            krank = chess.square_rank(KINGS[not color])
            kfile = chess.square_file(KINGS[not color])

            for square, piece in PIECES[color]:
                ptype = piece.piece_type

                # Attack cache
                attacks_cache[square] = board.attacks(square)

                # Mobility
                mobility_count = len(attacks_cache[square])
                weight = piece_weight[ptype]
                mobility_score += (mobility_count ** 0.75) * weight * 0.5

                # Heatmap
                rank = square >> 3
                file = square & 7

                heatmap_rank = 7 - rank if piece.color == chess.WHITE else rank
                heatmap_score += heatmap[ptype][heatmap_rank][file]

                # MPDB
                if ptype in [chess.KNIGHT, chess.BISHOP]:
                    if square not in starting_squares[color][ptype]:
                        minor_piece_development_bonus += 1.5

                # Aggression
                dist = max(abs(rank - krank), abs(file - kfile))
                aggression_score += self.MATERIAL[ptype] * -(dist ** 1.5)

            return {
                "mobility_score" : mobility_score,
                "heatmap_score" : heatmap_score,
                "minor_piece_development_bonus" : minor_piece_development_bonus,
                "aggression_score" : aggression_score,
                "attacks_cache" : attacks_cache
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

                attacked_square_counts = collections.Counter(attacked_squares)
                center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
                
                # Calculate cover bonus denominator once
                cover_denominator = sum(self.MATERIAL[piece_map[sq].piece_type] for sq in attacks_cache.keys())
                cover_bonus = len(attacked_squares) / cover_denominator if cover_denominator > 0 else 0

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

                    piece_weight_val = piece_weight[piece_type] ** (1/2)
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

                if stage != Computer.stageLATE and (not board.has_castling_rights(color)) and has_moved:
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

                for file_bb in pawn_files_bb:
                    if chess.popcount(file_bb) > 1:
                        pawn_score -= 1.5

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
                    pawn_score += 1.5

                # Passed pawns
                for square in chess.SquareSet(pawns_bb):
                    file = square & 7
                    rank = square >> 3

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
                        rank_bonus = (1 + (rank / 7 if color == chess.WHITE else 1 - rank / 7)) ** 4
                        pawn_score += (3.0 if stage == Computer.stageLATE else 2.0) * rank_bonus * num_protectors
                        if connected & (1 << square):
                            pawn_score += (4.0 if stage == Computer.stageLATE else 2.0) * rank_bonus * num_protectors # Extra bonus if pawn is connected
                    
                    # Give a small exponential bonus for pawns closer to the end to encourage pushing closer pawns
                    prom_rank = 7 if color == chess.WHITE else 0
                    distance = abs(prom_rank - rank)
                    pawn_score += 5 * distance ** -1.5 * num_protectors

                # Store pawn hash in cache
                self.pawn_hash_cache[pawn_hash] = pawn_score
                return pawn_score

            def attack_quality() -> float:

                aggression_score = PIECE_SCORES["aggression_score"]
                material_diff = material[color] - material[not color]

                if board.is_check():
                    if board.turn == color:
                        aggression_score -= 5
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
                    
                    # Skip if no victim or attacker (shouldn't happen since non-captures are skipped, 
                    # but it stops the errors showing up)
                    if not victim or not attacker:
                        continue

                    victim_material = self.MATERIAL[victim]
                    attacker_material = self.MATERIAL[attacker]
                    if victim_material > attacker_material:
                        aggression_score += 2.0
                    if material_diff + victim_material - attacker_material > 1.5:
                        aggression_score += 1.5
                    elif material_diff + victim_material - attacker_material < 0.75:
                        aggression_score -= 3.0

                for square, piece in PIECES[not color]:
                    attackers = board.attackers_mask(color, square)
                    len_attackers = chess.popcount(attackers)

                    if len_attackers == 0:
                        continue

                    ptype = piece.piece_type

                    defenders = board.attackers(not color, square)
                    if len(defenders) == 0:
                        aggression_score += self.MATERIAL[ptype] * len_attackers * 4.0
                    else:
                        defender = board.piece_type_at(next(iter(defenders)))
                        if defender is None:
                            continue
                        defender_value = self.MATERIAL[defender]
                        if self.MATERIAL[ptype] < defender_value:
                            aggression_score += self.MATERIAL[ptype] * len_attackers
                        else:
                            aggression_score -= self.MATERIAL[ptype] * len_attackers * 0.5

                return aggression_score

            score = 0
            score -= king_safety_penalty() * (0.75 if stage == Computer.stageLATE else 7.5)
            # score -= (25 * low_legal_penalty()) ** 1.5 * (aggression[not color] ** 2)
            score += material_score() ** 3 * (15 if stage == Computer.stageLATE else 5)
            score += cse(coverage(), 2) * aggression[color] * (5.5 if stage == Computer.stageLATE else 2)
            score += cse(control(), 2) * aggression[color] * (12 if stage == Computer.stageLATE else 6.5 if stage == Computer.stageEARLY else 4.5)
            score += heatmap() ** (3 if stage == Computer.stageEARLY else 2 if stage == Computer.stageLATE else 1) * aggression[not color] * (35 if stage == Computer.stageLATE else 22.5 if stage == Computer.stageEARLY else 15)
            score += minor_piece_bonus() * 15 * aggression[color]
            score += cse(pawn_structure(), 1.5) * (8.5 if stage == Computer.stageLATE else 3.75)
            score += cse(attack_quality(), 1.3) * aggression[color] * (3.75 if stage == Computer.stageLATE else 15)

            if isinstance(score, complex):
                print("\nAGG", aggression[color])
                print("EAG", aggression[not color])
                print("-KSP", king_safety_penalty() * 5.5)
                print("-LLP", (25 * low_legal_penalty()) ** 1.5 * (aggression[not color] ** 2))
                print("+MS", material_score() ** 2 * 20)
                print("+COV", coverage() * aggression[color] * (0.55 if stage == Computer.stageLATE else 0.2))
                print("+HEAT", heatmap() ** (3 if stage == Computer.stageEARLY else 1) * aggression[not color] * 3 * (10 if stage == Computer.stageLATE else 7.5 if stage == Computer.stageEARLY else 5))
                print("+CTRL", (control() ** 1.25) * aggression[color] * (1.2 if stage == Computer.stageLATE else 0.65 if stage == Computer.stageEARLY else 0.45))
                print("+MIN", minor_piece_bonus() * 15 * aggression[color])
                print("+PAWN", cse(pawn_structure(), 7/5) * 2.5 * (2 if stage == Computer.stageLATE else 1.5))
                print("+ATT", attack_quality() ** 1.2 * aggression[color] * 15)
                raise ValueError("Score is complex")

            return score

        score = 0
        score += evaluate_player(chess.WHITE)
        score -= evaluate_player(chess.BLACK)

        self.transposition_table[transposition_key] = score

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
            return opening_best
    
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

            # Get best score: If can_syzygy and score is not high enough (or is not completely lost) after a shallow search, then use syzygy
            if depth > 2:
                best_score = maxmin(move_score_map, key=lambda x: x[1])[1]
                move = self.best_syzygy(board, best_score)
                if move is not None:
                    print("Using Syzygy:",board.san(move))
                    return move

            # Gradually filter out based on the previous scores
            # Force computers to look to depth 3 & for at least 1 second
            if not reset and depth >= 3 and time.time() - self.start_time > 1:
                if time.time() - self.start_time <= (self.timeout * 0.1):
                    threshold = 0.75
                elif stage != Computer.stageLATE and depth <= 3:
                    threshold = 0.05 * (depth - 1) * (len(list(board.legal_moves)) / len(moves))
                    threshold = min(threshold, 0.35)
                else:
                    threshold = 0.1 * (depth - 1) * (len(list(board.legal_moves)) / len(moves))
                    threshold = min(threshold, 0.35)
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
                if depth > 3 and stage != Computer.stageLATE: # Only apply when the computer has a good idea of the position and not in the endgame
                    if i > self._turning_point([score for _, score in move_score_map], threshold=1.5 - (i / len(moves))): # Dynamic threshold based on move number
                        print("LMR",end='\t')
                        reduction += 1

                board.push(move)
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

        self.display_metrics()
        self.print_board_feedback(board, move_score_map)

        return best_move

    ##################################################
    #                     EXTRAS                     #
    ##################################################

    def get_game_stage(self, piece_map: dict) -> int:
        """Return the current stage of the game."""

        num_pieces = len([piece for piece in piece_map.values() if piece.piece_type != chess.PAWN])
        if num_pieces >= 12:
            return Computer.stageEARLY
        elif num_pieces >= 8:
            return Computer.stageMIDDLE
        else:
            return Computer.stageLATE

    def is_timeup(self) -> bool:
        if self.timeout is None or self.start_time is None:
            return False
        return time.time() - self.start_time > self.timeout

    def normalise_score(self, score: float) -> float:
        return round(score / self.ESTIMATED_PAWN_VALUE, 2)
    
    def display_metrics(self) -> None:
        
        elapsed = time.time() - self.start_time if self.start_time is not None else float("nan")
        cache_hit_proportion = self.cache_hits / self.leaf_nodes_explored if self.leaf_nodes_explored > 0 else float("nan")
        print(f"""
        Nodes explored      : {self.nodes_explored} | NPS : {self.nodes_explored / elapsed:.2f}
        Leaf nodes explored : {self.leaf_nodes_explored} | LNPS : {self.leaf_nodes_explored / elapsed:.2f}
        Prunes:
            Alpha : {self.alpha_cuts}
            Beta  : {self.beta_cuts}
            Total : {self.prunes}
        Cache hits : {self.cache_hits} | {cache_hit_proportion * 100:.2f}%
        Time elapsed: {elapsed:.2f}s
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
        score = self.normalise_score(scoremap[0][1])
        score = score if board.turn == chess.WHITE else -score

        print(f"""
{__version__.capitalize()}: I evaluate the position at {score} in my favour.
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
    # FEN = "3n4/8/5k2/8/4p3/1p2B3/1P2BPK1/8 w - - 0 74" # Test cleanup ability when clearly winning

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
