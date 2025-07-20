import argparse
import chess
import chess.pgn
import os
import sys

from datetime import datetime

sys.path.insert(0, 'chess')  # Adjust path to import other_bots module
import other_bots

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
    nargs=2,
    choices=player_map.keys(),
    help="Select the players to play with.",
    default=['main', 'main']
)
parser.add_argument(
    '-t', '--timeout', '--time',
    type=int,
    help="Set the timeout for each player in seconds.",
    default=10
)
parser.add_argument(
    '-l', '--log',
    action='store_true',
    help="Enable logging of the game moves to a file.",
    default=False
)
args = parser.parse_args()

players = [player_map[bot].Computer for bot in args.players]

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
    game.headers["Event"] = "Chess Bot Match"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime('%Y.%m.%d')
    game.headers["Round"] = "1"
    game.headers["White"] = args.players[0].capitalize()
    game.headers["Black"] = args.players[1].capitalize()
    game.headers["Result"] = board.result()

    # Set the FEN header if the board is not in the starting position
    if FEN != chess.STARTING_BOARD_FEN:
        game.headers["FEN"] = FEN
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

FEN = "8/3k4/1b5R/8/2K2B2/8/p7/8 w - - 0 1"

def main():

    MOVELOG_PATH = f"chess/play/logs/{SESSION}_{args.players[0].upper()}_{args.players[1].upper()}/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"

    if args.log:
        os.makedirs(os.path.dirname(MOVELOG_PATH), exist_ok=True)
        MOVELOG_FILE = open(MOVELOG_PATH, 'w')
    else:
        MOVELOG_FILE = sys.stdout

    print(f"Game log will be saved to: {MOVELOG_PATH}")

    def print_and_log(*print_args, **kwargs):
        """Prints to console and logs to file."""
        print(*print_args, **kwargs)
        if args.log:
            print(*print_args, file=MOVELOG_FILE, **kwargs)

    board = chess.Board(fen=FEN)

    print_and_log(header())

    while not board.is_game_over():

        for player in players:

            print(board, file=MOVELOG_FILE)
            print(board, "\n\n")

            if board.is_game_over():
                break

            move = player(board.turn).best_move(board, timeout=args.timeout)
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