import sys
sys.path.insert(0, 'chess_bot')

import argparse
import chess
import chess.engine
import torch
import numpy as np
import random as rnd
from tqdm import tqdm

from nnue.model import Net
from bot.main import Computer

def random_board() -> chess.Board:
    board = chess.Board()
    for i in range(0, np.random.randint(10, 200)):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = rnd.choice(legal_moves)
        board.push(move)
        if board.is_game_over():
            break
    return board

def test_nnue_strength(net_path: str, num_positions: int = 1000) -> float:
    # Load the net
    net = Net()
    try:
        net.load(net_path)
        print(f"Loaded net from {net_path}")
    except FileNotFoundError:
        print(f"Could not find net at {net_path}")
        exit(1)
    net.eval()

    # Create computer for hardcode evaluation
    C = Computer(chess.WHITE)

    nnue_scores = []
    stockfish_scores = []
    hardcode_scores = []
    with torch.no_grad():
        for i in tqdm(range(num_positions), desc="Testing NNUE"):
            board = random_board()

            # NNUE evaluation
            nnue_score = C.evaluate(board)
            nnue_normalised = C.normalise_score(nnue_score)
            nnue_scores.append(nnue_normalised)

            # Hardcode evaluation
            hardcode_score = C.hardcode_evaluate(board)
            score_normalised = C.normalise_score(hardcode_score)
            hardcode_scores.append(score_normalised)

            # Stockfish evaluation
            with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
                stockfish_result = engine.analyse(board, chess.engine.Limit(time=0.0001))
                if 'score' in stockfish_result:
                    stockfish_score = stockfish_result['score'].white().score(mate_score=100000)
                    stockfish_normalised = stockfish_score / 100.0  # Convert centipawns to pawns
                else:
                    stockfish_normalised = 0.0  # Default if no score
            stockfish_scores.append(stockfish_normalised)

    # Compute average ratio between NNUE and Stockfish scores
    ratios = []
    for nnue, sf in zip(nnue_scores, stockfish_scores):
        if sf != 0:
            ratios.append(nnue / sf)
    average_ratio = sum(ratios) / len(ratios) if ratios else 1.0

    total_error = 0.0
    total_stockfish_error = 0.0
    for i in range(num_positions):
        nnue_normalised = nnue_scores[i]
        score_normalised = hardcode_scores[i]
        stockfish_normalised = stockfish_scores[i]

        # Compute weighted error: errors matter less when values are far from 0, more when close to 0
        avg_value = (nnue_normalised + score_normalised) / 2
        error = abs(nnue_normalised - score_normalised) / (1 + abs(avg_value))
        total_error += error

        # Adjust stockfish score by average ratio and compute error
        adjusted_stockfish = stockfish_normalised * average_ratio
        stockfish_avg_value = (nnue_normalised + adjusted_stockfish) / 2
        stockfish_error = abs(nnue_normalised - adjusted_stockfish) / (1 + abs(stockfish_avg_value))
        total_stockfish_error += stockfish_error

        # Display 10 samples
        if i < 10:
            print(f"T {score_normalised:.2f} | E {nnue_normalised:.2f} | S {stockfish_normalised:.2f} | AS {adjusted_stockfish:.2f} | e {error:.2f} | se {stockfish_error:.2f}")

    average_error = total_error / num_positions
    average_stockfish_error = total_stockfish_error / num_positions
    print(f"Average ratio: {average_ratio:.4f}")
    print(f"Average stockfish evaluation error: {average_stockfish_error:.4f}")
    return average_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the strength of the NNUE network.")
    parser.add_argument('net_path', help='Path to the NNUE network file (e.g., net.pt)')
    parser.add_argument('-p', '--num_positions', type=int, default=1000, help='Number of positions to test (default: 1000)')
    args = parser.parse_args()

    score = test_nnue_strength(args.net_path, args.num_positions)
    print(f"Average evaluation error: {score:.4f}")
