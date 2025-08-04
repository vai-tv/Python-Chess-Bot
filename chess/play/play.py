#!/usr/bin/env python3
"""
Refactored Chess Game Manager
A clean class-based architecture for running chess games between bots.
"""

import argparse
import chess
import chess.pgn
import chess.engine
import os
import sys
import time
from collections import deque
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


class MoveLogger:
    """Handles logging of game moves and display formatting."""
    
    def __init__(self, log_file: Optional[Any] = None):
        self.log_file = log_file
        
    def print_and_log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file is not None:
            print(*args, file=self.log_file, **kwargs)
            
    def header(self, game_count: int, wins: List[float], players: List[str], timeout: float) -> str:
        index = game_count % 2
        return f"""
 ----------- {(f'{players[index].upper()} ({wins[index]})').ljust(10)} VS {(f'({wins[1 - index]}) {players[1 - index].upper()}').rjust(10)} ------------
| TIME : {(str(timeout) + ' SECONDS PER MOVE').rjust(40)} |
| GAME BEGINS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(30)} |
 -------------------------------------------------
"""

    def footer(self, board: chess.Board, winner: str, winner_name: str, 
               game_count: int, wins: List[float], players: List[str], fen: str) -> str:
        index = game_count % 2

        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Bot Match"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime('%Y.%m.%d')
        game.headers["Round"] = str(game_count + 1)
        game.headers["White"] = players[index].capitalize()
        game.headers["Black"] = players[1 - index].capitalize()
        game.headers["Result"] = board.result()

        if fen != chess.STARTING_BOARD_FEN:
            game.headers["FEN"] = fen
            
        node = game
        for move in board.move_stack:
            node = node.add_variation(move)
            
        return f"""
 ----------- {(f'{players[index].upper()} ({wins[index]})').ljust(10)} VS {(f'({wins[1 - index]}) {players[1 - index].upper()}').rjust(10)} ------------
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
            
            for i in range(self.opening_moves * 2):
                move = self.players[0](board_copy.turn).random_opening_move(board_copy)
                start = time.time()
                
                while move is None:
                    if time.time() - start > 10:
                        break
                    move = self.players[0](board_copy.turn).random_opening_move(board_copy)
                    
                if move is None:
                    print("No more theory.")
                    break
                print(f"Move {(i + 2) // 2}: {board_copy.san(move)}", end='\t')
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
                 opening_handler: Optional[OpeningHandler], timeout: float,
                 log_enabled: bool, players: List[str]):
        self.game_state = game_state
        self.move_logger = move_logger
        self.opening_handler = opening_handler
        self.timeout = timeout
        self.log_enabled = log_enabled
        self.players = players
        self.session = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{players[0].upper()}_{players[1].upper()}"
        self.TIME_AT_START = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    def setup_logging(self, game_count: int):
        if not self.log_enabled:
            return None
        
        self.TIME_AT_START = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        move_log_path = f"chess/play/logs/{self.session}/G{game_count + 1}_{self.TIME_AT_START}.log"
        os.makedirs(os.path.dirname(move_log_path), exist_ok=True)
        return open(move_log_path, 'w')
        
    def cleanup_logging(self, move_log_file, game_count: int):
        if move_log_file is not None and move_log_file != sys.stdout:
            move_log_file.close()
            move_log_path = f"chess/play/logs/{self.session}/G{game_count + 1}_{self.TIME_AT_START}.log"
            log_name = f"{move_log_path.split('.')[0]}_FINISHED.log"
            os.rename(move_log_path, log_name)
            
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
                
            return winner, winner_name
        else:
            self.game_state.wins[0] += 0.5
            self.game_state.wins[1] += 0.5
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
            
            fen = self.game_state.original_fen
            if self.opening_handler:
                fen = self.opening_handler.play_opening_moves(self.game_state, fen)
                self.game_state.board = chess.Board(fen)
                
            if move_log_file is not None:
                print(f"Game log will be saved to: chess/play/logs/{self.session}/G{game_count + 1}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log")
                
            self.move_logger.print_and_log(
                self.move_logger.header(game_count, self.game_state.wins, self.players, self.timeout)
            )
            
            while not self.game_state.is_game_over():
                for player_idx, player in enumerate(self.game_state.current_players):
                    self.move_logger.print_and_log(self.game_state.board, "\n\n")
                    
                    if self.game_state.is_game_over():
                        break
                        
                    current_player = player(self.game_state.get_turn())
                    move = current_player.best_move(self.game_state.board, timeout=self.timeout)
                    
                    if move is None:
                        return False
                        
                    if self.game_state.get_turn() == chess.WHITE:
                        self.move_logger.print_and_log(f"\n{self.game_state.board.fullmove_number}: {self.game_state.board.san(move)}")
                    else:
                        self.move_logger.print_and_log(f"... {self.game_state.board.san(move)}")
                        
                    self.game_state.push_move(move)
                    
            self.move_logger.print_and_log(self.game_state.board, "\n\n")
            
            result = self.game_state.get_result()
            winner, winner_name = self.determine_winner(result, game_count)
            
            self.move_logger.print_and_log(
                self.move_logger.footer(self.game_state.board, winner, winner_name, 
                                      game_count, self.game_state.wins, self.players, fen)
            )
            
            return True
            
        finally:
            self.cleanup_logging(move_log_file, game_count)
            
    def run(self):
        while True:
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
            help="Select the players to play with.",
            default=['main', 'main']
        )
        parser.add_argument(
            '-t', '--timeout', '--time',
            type=float,
            help="Set the timeout for each player in seconds.",
            default=10.0
        )
        parser.add_argument(
            '-l', '--log',
            action='store_true',
            help="Enable logging of the game moves to a file.",
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
            timeout=self.args.timeout,
            log_enabled=self.args.log,
            players=self.args.players
        )
        
        try:
            game_loop.run()
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            sys.exit(0)
        except Exception as e:
            # Log error using GameLoop's session
            session = game_loop.session
            players = self.args.players
            
            error_dir = f"chess/play/logs/{session}"
            os.makedirs(error_dir, exist_ok=True)
            
            with open(f"{error_dir}/error.log", "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {e.__class__.__name__}: {e}\n")
            
            print(f"Game interrupted by error: {e}")
            print(f"Error logs saved to: {error_dir}")
            
            time.sleep(5)


if __name__ == "__main__":
    manager = ChessGameManager()
    manager.run()
