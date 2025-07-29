import argparse
import chess
import chess.pgn
import chess.engine
import os
import sys
import time

from collections import deque

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
parser.add_argument(
    '-om', '--opening-moves',
    type=int,
    help="Number of opening moves to play before letting the bots play. Advised to be around 10.",
    default=0
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
    game.headers["Round"] = str(game_count + 1)
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

{pgn_string}"""

SESSION = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

####################################################################################################
#                                             MAIN LOOP                                            #
####################################################################################################

FEN = chess.STARTING_FEN  # Starting position FEN
WINS = [0, 0]  # Initialize win counts for players

def play_opening_moves(board: chess.Board) -> chess.Board:
    """Play the opening moves of the game by the logic of the white computer."""

    if args.opening_moves == 0:
        return board

    while True:
        
        board_copy = board.copy()

        for i in range(args.opening_moves * 2):
            
            move = players[0](board_copy.turn).random_opening_move(board_copy)
            start = time.time()

            while move is None:
                if time.time() - start > 10:
                    break
                move = players[0](board_copy.turn).random_opening_move(board_copy)
                
            if move is None:
                print("No more theory.")
                break
            print(f"Move {(i + 2) // 2}: {board_copy.san(move)}",end='\t')
            board_copy.push(move)

        # Only continue if the position is close to equal in score (0 ± 10)
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        evaluation = engine.analyse(board_copy, chess.engine.Limit(time=1))
        engine.quit()

        score = evaluation["score"].white().score() # type: ignore
        if score is None:
            continue
        if abs(score) < 10:
            print(f"\nPosition is close to equal, score is {score}. Continuing.")
            return board_copy
        
        print(f"\nPosition is not close to equal, score is {score}.")

def main():
    """
    Main function to play multiple games of chess with the given players.

    This function will play multiple games of chess, alternating the colors of the players
    each game. The FEN of the starting position is used for each game. The games are logged
    to a file if specified with the --log flag.

    The wins of each player are tracked and displayed after each game. The games are numbered
    from 1 to the number of games specified.

    The function will loop indefinitely until the user stops it.

    :return: None
    """

    def print_and_log(*print_args, **kwargs):
        """Prints to console and logs to file."""
        print(*print_args, **kwargs)
        if args.log:
            print(*print_args, file=MOVELOG_FILE, **kwargs)

    global players, game_count
    original_players = players.copy()
    game_count = 0

    while True:
        MOVELOG_PATH = f"chess/play/logs/{SESSION}_{args.players[0].upper()}_{args.players[1].upper()}/G{game_count + 1}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"

        if args.log:
            os.makedirs(os.path.dirname(MOVELOG_PATH), exist_ok=True)
            MOVELOG_FILE = open(MOVELOG_PATH, 'w')
        else:
            MOVELOG_FILE = sys.stdout

        print(f"Game log will be saved to: {MOVELOG_PATH}")

        # Assign players for this game based on game count to swap colors
        if game_count % 2 == 0:
            players = original_players
        else:
            players = original_players[::-1]

        board = chess.Board(fen=FEN)
        board = play_opening_moves(board)

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

        # Add FINISHED to the end of the file log name
        log_name = f"{MOVELOG_PATH.split('.')[0]}_FINISHED.log"
        os.rename(MOVELOG_PATH, log_name)

        time.sleep(2)


if __name__ == "__main__":
    error_timestamps = deque()
    error_threshold = 10

    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
            break
        except Exception as e:
            current_time = time.time()
            error_timestamps.append(current_time)

            # Remove timestamps older than 60 seconds
            while error_timestamps and current_time - error_timestamps[0] > 60:
                error_timestamps.popleft()

            if len(error_timestamps) >= error_threshold:
                print(f"{len(error_timestamps)} errors occurred within 1 minute. Pausing and rebooting.")
                time.sleep(60)
                # Rerun the file
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                print(f"An error occurred: {e.__class__.__name__} {e}")
                print("Restarting the game in three seconds...")
                time.sleep(3)
