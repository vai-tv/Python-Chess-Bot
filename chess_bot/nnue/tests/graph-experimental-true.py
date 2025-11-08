import argparse
import chess
import matplotlib.pyplot as plt
import random as rnd
import torch
import os

import sys
sys.path.insert(0, 'chess_bot')

from nnue.model import Net
from bot.main import Computer


def uniform_random_board():
    board = chess.Board()
    for _ in range(rnd.randint(100, 300)):
        legals = list(board.legal_moves)
        if not legals:
            break
        board.push(rnd.choice(legals))
    return board


def main(num_positions: int):
    nets_dir = "chess_bot/nnue/nets"
    net_files = [f for f in os.listdir(nets_dir) if f.endswith('.pt')]

    if not net_files:
        print("No net files found in", nets_dir)
        return

    fig, axes = plt.subplots(1, len(net_files) + 1, figsize=(6*(len(net_files) + 1), 6))
    if len(net_files) + 1 == 1:
        axes = [axes]

    # Generate n evaluation-board pairs
    C = Computer(chess.WHITE)
    boards = [uniform_random_board() for _ in range(num_positions)]
    hardcode_scores = [C.hardcode_evaluate(board) for board in boards]
    normalised_hc = [C.normalise_score(score) for score in hardcode_scores]

    all_nnue_scores = []

    for idx, net_file in enumerate(net_files):
        net_path = os.path.join(nets_dir, net_file)
        net = Net()
        try:
            net.load(net_path)
            print(f"Loaded net from {net_file}")
        except FileNotFoundError:
            print(f"Could not find net at {net_path}")
            continue
        net.eval()

        nnue_scores = []

        with torch.no_grad():
            for i in range(num_positions):
                board = boards[i]

                # NNUE evaluation
                nnue_score = C.evaluate(board)
                nnue_normalised = C.normalise_score(nnue_score)
                nnue_scores.append(nnue_normalised)

        all_nnue_scores.append((net_file, nnue_scores))

        ax = axes[idx]
        ax.scatter(normalised_hc, nnue_scores, alpha=0.5)
        ax.set_xlabel('Hardcode Evaluation')
        ax.set_ylabel('NNUE Prediction')
        ax.set_title(f'{net_file}')
        ax.grid(True)

    # Combined plot
    ax_combined = axes[-1]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # List of colors
    for i, (net_file, nnue_scores) in enumerate(all_nnue_scores):
        color = colors[i % len(colors)]
        ax_combined.scatter(normalised_hc, nnue_scores, alpha=0.5, color=color, label=net_file)
    ax_combined.set_xlabel('Hardcode Evaluation')
    ax_combined.set_ylabel('NNUE Prediction')
    ax_combined.set_title('Combined All Nets')
    ax_combined.legend()
    ax_combined.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate scatter plots comparing hardcode evaluation to NNUE predictions for each net.")
    parser.add_argument('-n', '--num_positions', type=int, default=1000, help='Number of random positions to evaluate (default: 1000)')
    args = parser.parse_args()

    main(args.num_positions)
