import chess

from bot.main import Computer
from other_bots.human import Human

players = [Computer,Human]

def main():

    board = chess.Board()

    while not board.is_game_over():

        for player in players:

            move = player(board.turn).best_move(board, timeout=10)
            board.push(move)
            print("Move:", move)
            print(board, "\n\n")

if __name__ == "__main__":
    main()