#!/usr/bin/env python3
"""
Refactored Chess Game Manager
A clean class-based architecture for running chess games between bots.
"""

import argparse
import chess
import chess.pgn
import chess.engine
import linecache
import os
import sys
import time
import traceback
from datetime import datetime
from typing import List, Optional, Any


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
        
    def reset(self):
        self.board = chess.Board(self.original_fen)
        
    def swap_players(self):
        if len(self.players) == 2:
            self.current_players = [self.players[1], self.players[0]]
        else:
            self.current_players = self.players.copy()
            
    def get_current_player(self, color: bool) -> Any:
        return self.current_players[0] if color == chess.WHITE else self.current_players[1]
        
    def is_game_over(self) -> bool:
        return self.board.is_game_over()
        
    def get_result(self) -> str:
        return self.board.result()
    
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
    
    def __init__(self, log_file: Optional[Any] = None):
        self.log_file = log_file
        
    def print_and_log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file is not None:
            print(*args, file=self.log_file, **kwargs)
            
    def header(self, game_count: int, wins: List[float], players: List[str], timeouts: List[float], elo_ratings: List[float]) -> str:
        white_info = f"{players[0].upper()} ({wins[0]} | {elo_ratings[0]:.0f})"
        black_info = f"({wins[1]} | {elo_ratings[1]:.0f}) {players[1].upper()}"
        return f"""
 ---- {white_info.ljust(17)} VS {black_info.rjust(17)} -----
| WHITE TIME: {(str(timeouts[0]) + 's').ljust(17)} {("BLACK TIME: " + str(timeouts[1]) + 's').rjust(17)} |
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
    
    def __init__(self, players: List[Any], opening_moves: int):
        self.players = players
        self.opening_moves = opening_moves
        
    def play_opening_moves(self, game_state: GameState, fen: str) -> str:
        if not hasattr(self.players[0], 'random_opening_move'):
            return fen
            
        if self.opening_moves == 0:
            return fen
            
        board = game_state.board.copy()
        
        while True:
            board_copy = board.copy()
            
            for _ in range(self.opening_moves * 2):
                move = self.players[0](board_copy.turn).random_opening_move(board_copy)
                start = time.time()
                
                while move is None:
                    if time.time() - start > 10:
                        break
                    move = self.players[0](board_copy.turn).random_opening_move(board_copy)
                    
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
                return board_copy.fen()
            print(f"\nPosition is not close to equal, score is {score}.")
            

class GameLoop:
    """Manages the main game execution loop."""
    
    def __init__(self, game_state: GameState, move_logger: MoveLogger, 
                 opening_handler: Optional[OpeningHandler], timeouts: List[float],
                 log_enabled: bool, players: List[str]):
        self.game_state = game_state
        self.move_logger = move_logger
        self.opening_handler = opening_handler
        self.timeouts = timeouts  # [player1_timeout, player2_timeout]
        self.log_enabled = log_enabled
        self.players = players
        self.session = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{players[0].upper()}_{players[1].upper()}"
        self.TIME_AT_START = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.opening_fen = game_state.original_fen
        
    def get_timeout_for_player_index(self, player_index: int) -> float:
        """Get the timeout for the player at the given index."""
        # Timeouts are always stored as [white_timeout, black_timeout]
        # When players are swapped, we need to return the appropriate timeout
        if player_index == 0:
            # First player in current_players - if game_count is even, this is white
            # if game_count is odd, this is black
            if self.game_state.game_count % 2 == 0:
                return self.timeouts[0]  # White's timeout
            else:
                return self.timeouts[1]  # Black's timeout
        else:
            # Second player in current_players - if game_count is even, this is black
            # if game_count is odd, this is white
            if self.game_state.game_count % 2 == 0:
                return self.timeouts[1]  # Black's timeout
            else:
                return self.timeouts[0]  # White's timeout
        
    def setup_logging(self, game_count: int):
        if not self.log_enabled:
            return None
        
        self.TIME_AT_START = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        move_log_path = f"chess/play/logs/{self.session}/G{game_count + 1}_{self.TIME_AT_START}.log"
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
                
            self.game_state.reset()
            
            self.opening_fen = self.game_state.original_fen
            if self.opening_handler:
                self.opening_fen = self.opening_handler.play_opening_moves(self.game_state, self.opening_fen)
                self.game_state.board = chess.Board(self.opening_fen)
                
            if move_log_file is not None:
                print(f"Game log will be saved to: chess/play/logs/{self.session}/G{game_count + 1}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log")
                
            self.move_logger.print_and_log(
                self.move_logger.header(game_count, self.game_state.wins, self.players, self.timeouts, self.game_state.elo_ratings)
            )
            
            while not self.game_state.is_game_over():
                for player in self.game_state.current_players:
                    material = self.game_state.calculate_material()
                    if self.game_state.get_turn() == chess.WHITE:
                        print("Material:",material)
                    else:
                        print("\nMaterial:",material)
                    print(self.game_state.board, "\n\n")
                    
                    if self.game_state.is_game_over():
                        break
                        
                    current_player = player(self.game_state.get_turn())
                    # Determine which player index this is and use their timeout
                    player_index = self.game_state.current_players.index(player)
                    move = current_player.best_move(self.game_state.board, timeout=self.get_timeout_for_player_index(player_index))
                    
                    if move is None:
                        return False
                        
                    if self.game_state.get_turn() == chess.WHITE:
                        self.move_logger.print_and_log(f"{self.game_state.board.fullmove_number}. {self.game_state.board.san(move).ljust(10)}",end='')
                    else:
                        self.move_logger.print_and_log(f"{self.game_state.board.san(move)}")
                        
                    self.game_state.push_move(move)
                    
            print(self.game_state.board, "\n\n")

            result = self.game_state.get_result()
            winner, winner_name = self.determine_winner(result, game_count)
            
            self.move_logger.print_and_log(
                self.move_logger.footer(self.game_state.board, winner, winner_name, 
                                      game_count, self.game_state.wins, self.players, self.opening_fen, self.game_state.elo_ratings)
            )

            move_log_path = f"chess/play/logs/{self.session}/G{game_count + 1}_{self.TIME_AT_START}.log"
            log_name = f"{move_log_path.split('.')[0]}_{winner_name}.log"
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
            nargs=2,
            type=float,
            help="Set the timeout for each player in seconds (player1 player2).",
            default=[10.0, 10.0]
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
        return parser.parse_args()
        
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
            with open(f"{error_dir}/G{game_loop.game_state.game_count + 1}_{game_loop.TIME_AT_START}.log", "a") as f:
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
        
        move_logger = MoveLogger()
        opening_handler = OpeningHandler(self.players, self.args.opening_moves) if self.args.opening_moves > 0 and self.args.fen == chess.STARTING_FEN else None
        
        game_loop = GameLoop(
            game_state=game_state,
            move_logger=move_logger,
            opening_handler=opening_handler,
            timeouts=self.args.timeout,
            log_enabled=self.args.log,
            players=self.args.players
        )
        
        while True:
            try:
                game_loop.run()
            except BaseException as e:
                self.log_error(e, game_loop)


if __name__ == "__main__":
    manager = ChessGameManager()
    manager.run()
