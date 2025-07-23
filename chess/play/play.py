import argparse
import chess
import chess.pgn
import os
import sys
import time

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
args = parser.parse_args()

players = [player_map[bot].Computer for bot in args.players]

print(f"Selected players: {args.players[0].capitalize()} vs {args.players[1].capitalize()}")

####################################################################################################
#                                             MOVE LOGS                                            #
####################################################################################################

def header():
    index = game_count % 2
    return f"""
 ----------- {(f'{args.players[index].upper()} ({WINS[index]})').ljust(10)} VS {(f'({WINS[1 - index]}) {args.players[1 - index].upper()}').rjust(10)} ------------
| TIME : {(str(args.timeout) + ' SECONDS PER MOVE').rjust(40)} |
| GAME BEGINS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(30)} |
 -------------------------------------------------
"""

def footer(board: chess.Board, winner: str, winner_name: str):
    index = game_count % 2

    # Create a PGN game from the board's move stack
    game = chess.pgn.Game()
    game.headers["Event"] = "Chess Bot Match"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime('%Y.%m.%d')
    game.headers["Round"] = "1"
    game.headers["White"] = args.players[index].capitalize()
    game.headers["Black"] = args.players[1 - index].capitalize()
    game.headers["Result"] = board.result()

    # Set the FEN header if the board is not in the starting position
    if FEN != chess.STARTING_BOARD_FEN:
        game.headers["FEN"] = FEN
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    pgn_string = str(game)

    return f"""
 ----------- {(f'{args.players[index].upper()} ({WINS[index]})').ljust(10)} VS {(f'({WINS[1 - index]}) {args.players[1 - index].upper()}').rjust(10)} ------------
| GAME ENDS AT : {datetime.now().strftime('%Y-%m-%d %H:%M:%S').rjust(32)} |
| WINNER : {f'{winner_name} ({winner.upper()})'.rjust(38)} |
 -------------------------------------------------

{pgn_string}
"""

SESSION = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

####################################################################################################
#                                             MAIN LOOP                                            #
####################################################################################################

FEN = chess.STARTING_FEN  # Starting position FEN
WINS = [0, 0]  # Initialize win counts for players

def main():
    global players, game_count
    original_players = players.copy()
    game_count = 0

    while True:
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

        # Assign players for this game based on game count to swap colors
        if game_count % 2 == 0:
            players = original_players
        else:
            players = original_players[::-1]

        board = chess.Board(fen=FEN)

        print_and_log(header())

        while not board.is_game_over():

            for player in players:

                print_and_log(board, "\n\n")

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

        print_and_log(board, "\n\n")

        winner = "DRAW"
        winner_name = "DRAW"
        if board.is_game_over() and board.result() != '1/2-1/2':
            # Determine winner color
            winner_color = chess.WHITE if board.result() == '1-0' else chess.BLACK
            winner = "WHITE" if winner_color == chess.WHITE else "BLACK"

            # Increment wins
            win_index = 0 if winner == "WHITE" else 1
            if game_count % 2 == 0:
                WINS[win_index] += 1
                winner_name = args.players[win_index].upper()
            else:
                WINS[1 - win_index] += 1
                winner_name = args.players[1 - win_index].upper()

        print_and_log(footer(board, winner, winner_name))

        game_count += 1

        time.sleep(2)

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e.__class__.__name__} {e}")
            print("Restarting the game...")