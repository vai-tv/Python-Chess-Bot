#!/usr/bin/env python3
"""
Refactored Chess Game Manager
A clean class-based architecture for running chess games between bots.
"""

import argparse
import chess
import chess.pgn
import chess.engine
import json
import linecache
import os
import random as rnd
import requests
import sys
import time
import traceback
import urllib.parse

from datetime import datetime

from typing import List, Optional, Any, Tuple

from packaging import version


class GameState:
    """Manages the current state of the chess game."""

    def __init__(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self.original_fen = fen
        self.players = []
        self.wins = [0.0, 0.0]  # White, Black
        self.game_count = 0
        self.current_players = []
        self.elo_ratings = [1500.0, 1500.0]  # Initialize ELO ratings for both players
        self.remaining_time = [0.0, 0.0]  # [white_remaining, black_remaining] in seconds
        self.time_bonus = [0.0, 0.0]  # [white_bonus, black_bonus] in seconds
        self.time_per_move = False  # Flag for time per move mode
        
    def reset(self, timeouts: List[Tuple[float, float]]):
        self.board = chess.Board(self.original_fen)
        # Initialize remaining time from base time and bonus
        self.remaining_time[0] = timeouts[0][0]  # White base time
        self.remaining_time[1] = timeouts[1][0]  # Black base time
        self.time_bonus[0] = timeouts[0][1]  # White bonus time
        self.time_bonus[1] = timeouts[1][1]  # Black bonus time
        
    def update_time_after_move(self, player_index: int, move_time: float, timeouts: List[Tuple[float, float]]):
        """Update remaining time after a move, subtracting move time and adding bonus."""
        if self.time_per_move:
            # Reset to base time for next move
            self.remaining_time[player_index] = timeouts[player_index][0]
        else:
            # Subtract the time used for the move
            self.remaining_time[player_index] -= move_time
            # Add bonus time
            self.remaining_time[player_index] += self.time_bonus[player_index]
        
    def swap_players(self):
        if len(self.players) == 2:
            self.current_players = [self.players[1], self.players[0]]
        else:
            self.current_players = self.players.copy()
            
    def get_current_player(self, color: bool) -> Any:
        return self.current_players[0] if color == chess.WHITE else self.current_players[1]
        
    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)
        
    def get_result(self) -> str:
        # Check for timeout only if time is not infinite
        if self.remaining_time[0] != float('inf') and self.remaining_time[0] <= 0:
            # if self.board.has_insufficient_material(chess.BLACK):
                return '1/2-1/2'
            # return '0-1'
        elif self.remaining_time[1] != float('inf') and self.remaining_time[1] <= 0:
            # if self.board.has_insufficient_material(chess.WHITE):
                return '1/2-1/2'
            # return '1-0'
        return self.board.result(claim_draw=True)
    
    def get_turn(self) -> bool:
        return self.board.turn
        
    def push_move(self, move: chess.Move):
        self.board.push(move)

    def calculate_material(self) -> int:
        material = 0
        material_table = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        for piece in self.board.piece_map().values():
            if piece.color == chess.WHITE:
                material += material_table[piece.piece_type]
            else:
                material -= material_table[piece.piece_type]
        return material

    def calculate_elo(self, result: str, game_count: int) -> dict:
        """
        Calculate new ELO ratings based on game result.
        Returns a dictionary with player names as keys and new ELO ratings as values.
        """

        color_index = game_count % 2

        K = 32  # Development factor
        
        if result == '1-0':  # White wins
            S_white, S_black = 1, 0
        elif result == '0-1':  # Black wins
            S_white, S_black = 0, 1
        else:  # Draw
            S_white, S_black = 0.5, 0.5
        
        # Get current ELO ratings
        R_white = self.elo_ratings[color_index]
        R_black = self.elo_ratings[1 - color_index]
        
        # Calculate expected scores
        E_white = 1 / (1 + 10 ** ((R_black - R_white) / 400))
        E_black = 1 / (1 + 10 ** ((R_white - R_black) / 400))
        
        # Calculate new ELO ratings
        new_white_elo = R_white + K * (S_white - E_white)
        new_black_elo = R_black + K * (S_black - E_black)
        
        # Update the ELO ratings
        self.elo_ratings[color_index] = new_white_elo
        self.elo_ratings[1 - color_index] = new_black_elo
        
        return {
            "white": new_white_elo,
            "black": new_black_elo,
            "white_expected": E_white,
            "black_expected": E_black
        }


class MoveLogger:
    """Handles logging of game moves and display formatting."""

    def __init__(self, log_file: Optional[Any] = None, time_per_move: bool = False):
        self.log_file = log_file
        self.time_per_move = time_per_move
        
    def print_and_log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file is not None:
            print(*args, file=self.log_file, **kwargs)
            
    def header(self, wins: List[float], players: List[str], timeouts: List[Tuple[float, float]], elo_ratings: List[float]) -> str:
        white_info = f"{players[0].upper()} ({wins[0]} | {elo_ratings[0]:.0f})"
        black_info = f"({wins[1]} | {elo_ratings[1]:.0f}) {players[1].upper()}"
        # Extract base and bonus times for display
        white_base, white_bonus = timeouts[0]
        black_base, black_bonus = timeouts[1]

        # Handle infinite timeout display
        def format_time_value(time_val: float) -> str:
            if time_val == float('inf'):
                return "inf"
            if round(time_val) == time_val:
                return str(int(time_val))
            return str(time_val)
        
        white_base_str = format_time_value(white_base)
        white_bonus_str = format_time_value(white_bonus)
        black_base_str = format_time_value(black_base)
        black_bonus_str = format_time_value(black_bonus)
        
        if self.time_per_move and white_bonus == 0:
            white_time_str = f"{white_base_str} PM"
        else:
            white_time_str = f"{white_base_str}+{white_bonus_str}"
        if self.time_per_move and black_bonus == 0:
            black_time_str = f"{black_base_str} PM"
        else:
            black_time_str = f"{black_base_str}+{black_bonus_str}"
        
        return f"""
 ---- {white_info.ljust(17)} VS {black_info.rjust(17)} -----
| WHITE TIME: {white_time_str.ljust(17)} {("BLACK TIME: " + black_time_str).rjust(17)} |
| GAME BEGINS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(30)} |
 -------------------------------------------------
"""

    def footer(self, board: chess.Board, winner: str, winner_name: str, 
               game_count: int, wins: List[float], players: List[str], fen: str, elo_ratings: List[float]) -> str:
        index = game_count % 2
        white_info = f"{players[0].upper()} ({wins[0]} | {elo_ratings[0]:.0f})"
        black_info = f"({wins[1]} | {elo_ratings[1]:.0f}) {players[1].upper()}"

        # Create a fresh board from the original FEN to ensure consistency
        fresh_board = chess.Board(fen)
        
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Bot Match"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime('%Y.%m.%d')
        game.headers["Round"] = str(game_count + 1)
        game.headers["White"] = players[index].capitalize()
        game.headers["Black"] = players[1 - index].capitalize()
        game.headers["WhiteElo"] = str(round(elo_ratings[index]))
        game.headers["BlackElo"] = str(round(elo_ratings[1 - index]))
        game.headers["Result"] = board.result()

        if fen != chess.STARTING_BOARD_FEN:
            game.headers["FEN"] = fen
            
        # Build the game by applying moves to the fresh board
        node = game
        for move in board.move_stack:
            # Verify the move is legal before adding it
            if move in fresh_board.legal_moves:
                fresh_board.push(move)
                node = node.add_variation(move)
            else:
                # If we encounter an illegal move, just skip it
                break
                
        return f"""
        
 ---- {white_info.ljust(17)} VS {black_info.rjust(17)} -----
| GAME ENDS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(32)} |
| WINNER : {f'{winner_name} ({winner.upper()})'.rjust(38)} |
 -------------------------------------------------

{str(game)}"""


class OpeningHandler:
    """Handles opening moves with Stockfish evaluation."""

    opening_path = os.path.join(os.path.dirname(__file__), "openings.json")

    if not os.path.exists(opening_path):
        with open(opening_path, "w") as f:
            json.dump({}, f)

    openings: dict[str, list[str]] = json.load(open(opening_path))
    
    def __init__(self, players: List[Any], opening_moves: int):
        self.players = players
        self.opening_moves = opening_moves

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

        url = "https://explorer.lichess.ovh/master?fen=" + fen_encoded

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2)
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

        response = self.opening_query(board)
        if "moves" in response and response["moves"]:
            moves = response["moves"]

            weights = [move["white"] + move["black"] + move["draws"] for move in moves]
            chosen_move = rnd.choices(moves, weights=weights, k=1)[0]
            return chess.Move.from_uci(chosen_move["uci"])
        return None
        
    def play_opening_moves(self, game_state: GameState, fen: str) -> str:
        if self.opening_moves == 0:
            return fen
        
        if str(self.opening_moves) not in self.openings:
            self.openings[str(self.opening_moves)] = []
        openings = self.openings[str(self.opening_moves)]

        # 1 / x
        # Exponential decay formula to discourage new openings when there are already many
        new_opening_chance = 1 / max(len(openings), 1)
        
        # Get opening from opening book
        if rnd.random() > new_opening_chance and openings:
            opening = rnd.choice(openings)
            print(f"Random float is above {(new_opening_chance * 100):.2f}%, reading from book.")
            return opening
        else:
            print(f"Random float is below {(new_opening_chance * 100):.2f}%, creating new opening.")
            
        board = game_state.board.copy()
        
        while True:
            board_copy = board.copy()
            
            for _ in range(self.opening_moves * 2):
                move = self.random_opening_move(board_copy)
                    
                if move is None:
                    print("No more theory.",end='\t', flush=True)
                    break
                print(f"{board_copy.san(move)}", end='\t', flush=True)
                board_copy.push(move)
                
            engine = chess.engine.SimpleEngine.popen_uci("stockfish")
            evaluation = engine.analyse(board_copy, chess.engine.Limit(time=1))
            engine.quit()
            
            score = evaluation["score"].white().score() # type: ignore
            if score is None:
                continue
            if abs(score) < 20:
                print(f"\nPosition is close to equal, score is {score}. Continuing.")

                # Save to the opening book
                opening = board_copy.fen()
                openings.append(opening)
                self.openings[str(self.opening_moves)] = openings

                json.dump(self.openings, open(self.opening_path, "w"), indent=4)

                return board_copy.fen()
            print(f"\nPosition is not close to equal, score is {score}.")
            

class GameLoop:
    """Manages the main game execution loop."""
    
    def __init__(self, game_state: GameState, move_logger: MoveLogger,
                 opening_handler: Optional[OpeningHandler], timeouts: List[Tuple[float, float]],
                 log_enabled: bool, players: List[str], time_per_move: bool = False):
        self.game_state = game_state
        self.move_logger = move_logger
        self.opening_handler = opening_handler
        self.timeouts = timeouts  # [(player1_base, player1_bonus), (player2_base, player2_bonus)]
        self.log_enabled = log_enabled
        self.players = players
        self.session = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{players[0].upper()}_{players[1].upper()}"
        self.TIME_AT_START = datetime.now().strftime('%d-%m %H:%M:%S')
        self.opening_fen = game_state.original_fen
        self.game_state.time_per_move = time_per_move
        self.move_logger.time_per_move = time_per_move
        
    def get_timeout_for_player_index(self, player_index: int) -> float:
        """Get the timeout for the player at the given index."""
        # Determine if the player at this index is white or black
        is_white = (self.game_state.game_count % 2 == 0 and player_index == 0) or \
                  (self.game_state.game_count % 2 == 1 and player_index == 1)
        player_color_index = 0 if is_white else 1  # 0 = white, 1 = black
        
        # Return the remaining time for the player (can be above base+bonus due to accumulated bonus)
        return self.game_state.remaining_time[player_color_index]
        
    def setup_logging(self, game_count: int):
        if not self.log_enabled:
            return None
        
        self.TIME_AT_START = datetime.now().strftime('%d-%m %H:%M:%S')
        move_log_path = f"chess/play/logs/{self.session}/G{game_count + 1} {self.TIME_AT_START}.log"
        os.makedirs(os.path.dirname(move_log_path), exist_ok=True)
        return open(move_log_path, 'w')
        
    def cleanup_logging(self, move_log_file):
        if move_log_file is not None and move_log_file != sys.stdout:
            move_log_file.close()
            
    def determine_winner(self, result: str, game_count: int) -> tuple[str, str]:
        if result != '1/2-1/2':
            winner_color = chess.WHITE if result == '1-0' else chess.BLACK
            winner = "WHITE" if winner_color == chess.WHITE else "BLACK"
            
            win_index = 0 if winner == "WHITE" else 1
            if game_count % 2 == 0:
                self.game_state.wins[win_index] += 1
                winner_name = self.players[win_index].upper()
            else:
                self.game_state.wins[1 - win_index] += 1
                winner_name = self.players[1 - win_index].upper()
                
            # Calculate and update ELO ratings
            elo_data = self.game_state.calculate_elo(result, game_count)
            
            # Display ELO update information
            print(f"\nELO Update:")
            print(f"White ({self.players[0]}): {elo_data['white']:.1f} (expected: {elo_data['white_expected']:.3f})")
            print(f"Black ({self.players[1]}): {elo_data['black']:.1f} (expected: {elo_data['black_expected']:.3f})")
            
            return winner, winner_name
        else:
            self.game_state.wins[0] += 0.5
            self.game_state.wins[1] += 0.5
            
            # Calculate and update ELO ratings for draw
            elo_data = self.game_state.calculate_elo(result, game_count)
            
            # Display ELO update information
            print(f"\nELO Update (Draw):")
            print(f"White ({self.players[0]}): {elo_data['white']:.1f} (expected: {elo_data['white_expected']:.3f})")
            print(f"Black ({self.players[1]}): {elo_data['black']:.1f} (expected: {elo_data['black_expected']:.3f})")
            
            return "DRAW", "DRAW"
            
    def play_single_game(self, game_count: int) -> bool:
        move_log_file = self.setup_logging(game_count)
        self.move_logger.log_file = move_log_file
        
        try:
            if game_count % 2 == 0:
                self.game_state.current_players = self.game_state.players
            else:
                self.game_state.current_players = [self.game_state.players[1], self.game_state.players[0]]
                
            # Reset the game state with timeouts
            self.game_state.reset(self.timeouts)
            
            self.opening_fen = self.game_state.original_fen
            if self.opening_handler:
                self.opening_fen = self.opening_handler.play_opening_moves(self.game_state, self.opening_fen)
                self.game_state.board = chess.Board(self.opening_fen)
                
            if move_log_file is not None:
                print(f"Game log will be saved to: chess/play/logs/{self.session}/G{game_count + 1} {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log")
                
            self.move_logger.print_and_log(
                self.move_logger.header(self.game_state.wins, self.players, self.timeouts, self.game_state.elo_ratings)
            )

            game_over = False
            
            while not self.game_state.is_game_over() and not game_over:

                for player in self.game_state.current_players:
                    # Material
                    material = self.game_state.calculate_material()
                    print("\nMaterial:",material)
                    print(self.game_state.board, "\n\n")
                    
                    if self.game_state.is_game_over():
                        break
                        
                    current_player = player(self.game_state.get_turn())
                    # Determine which player index this is and use their timeout
                    if self.game_state.get_turn() == chess.WHITE:
                        color = chess.WHITE if game_count % 2 == 0 else chess.BLACK
                    else:
                        color = chess.BLACK if game_count % 2 == 0 else chess.WHITE
                    player_index = 1 if color == chess.BLACK else 0
                    

                    # Get timeout
                    # Get the player module and check its version
                    player_module = sys.modules[current_player.__module__]
                    uses_old_timeout = version.parse(player_module.__version__) < version.parse('1.7.3')
                    if not self.game_state.time_per_move and uses_old_timeout: # First version which supports modern timeout
                        timeout = min(15, self.get_timeout_for_player_index(player_index) * 0.1) # Fallback
                        print(f"WARNING: Version {player_module.__version__} does not support modern timeout. Using fallback timeout of {timeout} seconds.")
                    else:
                        timeout = self.get_timeout_for_player_index(player_index)

                    # Start timing the move
                    move_start_time = time.time()

                    if uses_old_timeout and self.game_state.time_per_move:
                        move = current_player.best_move(self.game_state.board, timeout=timeout)
                    elif self.game_state.time_per_move:
                        move = current_player.best_move(self.game_state.board, time_per_move=timeout)
                    else:
                        move = current_player.best_move(self.game_state.board, timeout=timeout)
                    
                    move_end_time = time.time()
                    move_time = move_end_time - move_start_time

                    # Resign if time runs out (only check if time is not infinite)
                    if self.game_state.remaining_time[player_index] != float('inf') and self.game_state.remaining_time[player_index] <= 0:
                        game_over = True
                        break

                    if move is None:
                        game_over = True
                        break
                        
                    # Update the player's remaining time based on color, not index
                    self.game_state.update_time_after_move(player_index, move_time, self.timeouts)
                    
                    # Log time information - show both players' remaining time for debugging
                    self.move_logger.print_and_log(f"{self.game_state.board.fullmove_number}. {self.game_state.board.san(move).ljust(10)}",end='')
                    self.move_logger.print_and_log(f" (Time: {move_time:.2f}s, White: {self.game_state.remaining_time[0]:.2f}s, Black: {self.game_state.remaining_time[1]:.2f}s)")
                        
                    self.game_state.push_move(move)
                    
            print(self.game_state.board, "\n\n")

            result = self.game_state.get_result()
            winner, winner_name = self.determine_winner(result, game_count)
            
            self.move_logger.print_and_log(
                self.move_logger.footer(self.game_state.board, winner, winner_name, 
                                      game_count, self.game_state.wins, self.players, self.opening_fen, self.game_state.elo_ratings)
            )

            move_log_path = f"chess/play/logs/{self.session}/G{game_count + 1} {self.TIME_AT_START}.log"
            timeout_string = "-TO" if game_over else ""
            log_name = f"{move_log_path.split('.')[0]} {winner_name}{timeout_string}.log"
            os.rename(move_log_path, log_name)
            
            return True
            
        finally:
            self.cleanup_logging(move_log_file)
            
    def run(self):
        success = self.play_single_game(self.game_state.game_count)
        if success:
            self.game_state.game_count += 1
        time.sleep(2)


class ChessGameManager:
    """Main orchestrator for chess games."""
    
    def __init__(self):
        self.args = self._parse_arguments()
        self.setup_paths()
        self.players = self._setup_players()
        
    def _parse_timeout_format(self, timeout: list[str]) -> List[Tuple[float, float]]:
        """
        Parse timeout in format ["a+b", "c+d"] where:
        - a is base time for player 1
        - b is bonus time for player 1
        - c is base time for player 2  
        - d is bonus time for player 2
        
        Also supports backward compatibility with "a c" format.
        Supports "inf" for infinite timeout.
        """
        
        def parse_timeout_string(timeout_str: str) -> Tuple[float, float]:
            """Parse a single timeout string, handling 'inf' values."""
            if "+" in timeout_str:
                base_str, bonus_str = timeout_str.split('+')
                base = float('inf') if base_str.lower() == 'inf' else float(base_str)
                bonus = float('inf') if bonus_str.lower() == 'inf' else float(bonus_str)
                return (base, bonus)
            else:
                base = float('inf') if timeout_str.lower() == 'inf' else float(timeout_str)
                return (base, 0.0)
        
        timeoutstr1, timeoutstr2 = timeout
        timeout1 = [parse_timeout_string(timeoutstr1)]
        timeout2 = [parse_timeout_string(timeoutstr2)]
        return timeout1 + timeout2

    
    def _parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Play a chess game with selected bots.")
        sys.path.insert(0, 'chess')
        import other_bots
        player_map = {module.__name__.split('.')[-1]: module for module in other_bots.__all__}
        
        parser.add_argument(
            '-p', '--players',
            '--player',
            nargs=2,
            choices=list(player_map.keys()),
            help="Select the players to play with (player1 player2).",
            default=['main', 'main']
        )
        parser.add_argument(
            '-t', '--timeout', '--time',
            type=str,
            nargs=2,
            help="Set the timeout for each player in format 'a+b c+d' (base+bonus for each player) or 'a c' for backward compatibility. 'inf' to disable timeout.",
            default=["inf", "inf"]
        )
        parser.add_argument(
            '-tpm', '--timepermove',
            type=str,
            nargs=2,
            help="Set the time per move for each player, overriding the standard timeout. Use 'inf' for infinite time.",
            default=None
        )
        parser.add_argument(
            '-l', '--log',
            action='store_true',
            help="Enable logging of the game moves to a file when provided.",
            default=False
        )
        parser.add_argument(
            '-om', '--opening-moves',
            type=int,
            help="Number of opening moves to play before letting the bots play.",
            default=0
        )
        parser.add_argument(
            '-fen', '--fen',
            type=str,
            help="Initial FEN position of the board. Overrides opening moves if provided.",
            default=chess.STARTING_FEN
        )
        
        args = parser.parse_args()
        
        # Parse timeout string into the new format
        args.timeout = self._parse_timeout_format(args.timeout)

        # Handle time per move override
        if args.timepermove is not None:
            # Parse time per move values, allowing 'inf'
            tpm1 = float('inf') if args.timepermove[0].lower() == 'inf' else float(args.timepermove[0])
            tpm2 = float('inf') if args.timepermove[1].lower() == 'inf' else float(args.timepermove[1])
            # Set timeouts to time per move values with zero bonus
            args.timeout = [(tpm1, 0.0), (tpm2, 0.0)]
            args.time_per_move = True
        else:
            args.time_per_move = False
        
        return args
        
    def setup_paths(self):
        """Setup system paths for imports."""
        sys.path.insert(0, 'chess')
        
    def _setup_players(self) -> List[Any]:
        """Setup player modules."""
        import other_bots
        player_map = {module.__name__.split('.')[-1]: module for module in other_bots.__all__}
        return [player_map[bot].Computer for bot in self.args.players]

    def log_error(self, e: BaseException, game_loop: GameLoop) -> None:
        # Get traceback info
        tb = traceback.extract_tb(e.__traceback__)[-1]
        filename = tb.filename
        line_number = tb.lineno or 0
        line_content = linecache.getline(filename, line_number).strip()
        
        # Log error using GameLoop's session
        session = game_loop.session
        
        error_dir = f"chess/play/logs/{session}"
        if self.args.log:
            os.makedirs(error_dir, exist_ok=True)

        error_message = f"""{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ERROR: {e.__class__.__name__}
LOCATION: {filename}:{line_number}
LINE: {line_content}
TRACEBACK:

{traceback.format_exc()}
""" if not isinstance(e, KeyboardInterrupt) else "Game interrupted by user."

        footer = game_loop.move_logger.footer(game_loop.game_state.board, "ERROR", "ERROR", 
                                      game_loop.game_state.game_count, game_loop.game_state.wins, game_loop.players, game_loop.opening_fen, game_loop.game_state.elo_ratings)

        if self.args.log:
            if not isinstance(e, KeyboardInterrupt):
                with open(f"{error_dir}/error.log", "a") as f:
                    f.write(error_message + "\n\n")
            with open(f"{error_dir}/G{game_loop.game_state.game_count + 1} {game_loop.TIME_AT_START}.log", "a") as f:
                f.write(footer)
                f.write("\n\n" + error_message)

        if not isinstance(e, Exception):
                sys.exit(0)

        print(f"Game interrupted by error. Error logged to {error_dir}.")
        
        time.sleep(5)
        
    def run(self):
        """Main entry point."""
        print(f"Selected players: {self.args.players[0].capitalize()} vs {self.args.players[1].capitalize()}")
        
        game_state = GameState(fen=self.args.fen)
        game_state.players = self.players
        
        move_logger = MoveLogger(time_per_move=self.args.time_per_move)
        opening_handler = OpeningHandler(self.players, self.args.opening_moves) if self.args.opening_moves > 0 and self.args.fen == chess.STARTING_FEN else None
        
        game_loop = GameLoop(
            game_state=game_state,
            move_logger=move_logger,
            opening_handler=opening_handler,
            timeouts=self.args.timeout,
            log_enabled=self.args.log,
            players=self.args.players,
            time_per_move=self.args.time_per_move
        )
        
        while True:
            try:
                game_loop.run()
            except BaseException as e:
                self.log_error(e, game_loop)


if __name__ == "__main__":
    manager = ChessGameManager()
    manager.run()
