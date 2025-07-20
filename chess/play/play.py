import argparse
import chess
import chess.pgn
import os
import sys

from datetime import datetime

sys.path.insert(0, 'chess')  # Adjust path to import other_bots module
import other_bots
from bot import main as main_bot

####################################################################################################
#                                             GLOBALS                                              #
####################################################################################################

players = [module.Computer for module in other_bots.__all__ if hasattr(module, 'Computer')]
player_map = {module.__name__.split('.')[-1]: module for module in other_bots.__all__}

# Argument Parser to select players
parser = argparse.ArgumentParser(description="Play a chess game with selected bots.")
parser.add_argument(
    '-p', '--players',
    '--player',
    nargs='+',
    choices=player_map.keys(),
    help="Select the players to play with.",
    default=['random', 'random']
)
parser.add_argument(
    '-t', '--timeout', '--time',
    type=int,
    help="Set the timeout for each player in seconds.",
    default=10
)
args = parser.parse_args()

players = [player_map[bot].Computer for bot in args.players]

if len(players) != 2:
    raise ValueError("You must select exactly two players.")

print(f"Selected players: {args.players[0].capitalize()} vs {args.players[1].capitalize()}")

####################################################################################################
#                                             MOVE LOGS                                            #
####################################################################################################

def header():
    return f"""
 ----------- {args.players[0].upper().ljust(10)} VS {args.players[1].upper().rjust(10)} ------------
| TIME : {(str(args.timeout) + ' SECONDS PER MOVE').rjust(40)} |
| GAME BEGINS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(30)} |
 -------------------------------------------------
"""

def footer(board: chess.Board, winner: str):
    # Create a PGN game from the board's move stack
    game = chess.pgn.Game()
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    pgn_string = str(game)

    return f"""
 ----------- {args.players[0].upper().ljust(10)} VS {args.players[1].upper().rjust(10)} ------------
| TIME : {(str(args.timeout) + ' SECONDS PER MOVE').rjust(40)} |
| GAME ENDS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(32)} |
| WINNER : {winner.upper().rjust(38)} |
 -------------------------------------------------

{pgn_string}
"""

SESSION = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

####################################################################################################
#                                             MAIN LOOP                                            #
####################################################################################################

def main():

    MOVELOG_PATH = f"chess/play/logs/SESSION_{SESSION}/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{args.players[0].upper()}_{args.players[1].upper()}.log"

    os.makedirs(os.path.dirname(MOVELOG_PATH), exist_ok=True)
    MOVELOG_FILE = open(MOVELOG_PATH, 'w')

    print(f"Game log will be saved to: {MOVELOG_PATH}")

    def print_and_log(*args, **kwargs):
        """Prints to console and logs to file."""
        print(*args, **kwargs)
        print(*args, file=MOVELOG_FILE, **kwargs)

    board = chess.Board()

    print_and_log(header())

    while not board.is_game_over():

        for player in players:

            print(board, file=MOVELOG_FILE)
            print(board, "\n\n")

            if board.is_game_over():
                break

            move = player(board.turn).best_move(board, timeout=10)
            if move is None:
                return
            
            if board.turn == chess.WHITE:
                print_and_log(f"\n{board.fullmove_number}: {board.san(move)}")
            else:
                print_and_log(f"... {board.san(move)}")

            board.push(move)

    print(board, file=MOVELOG_FILE)
    print(board, "\n\n")

    winner = "White" if board.result() == "1-0" else "Black" if board.result() == "0-1" else "Draw"

    print_and_log(footer(board, winner))

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting the game...")