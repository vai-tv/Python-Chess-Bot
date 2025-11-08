import argparse
import chess
import matplotlib.pyplot as plt
import torch
import os

import sys
sys.path.insert(0, 'chess_bot')

from nnue.bootstrap import random_board
from nnue.model import Net
from bot.main import Computer


def main(num_positions: int):
    nets_dir = "chess_bot/nnue/nets"
    net_files = [f for f in os.listdir(nets_dir) if f.endswith('.pt')]

    if not net_files:
        print("No net files found in", nets_dir)
        return

    fig, axes = plt.subplots(1, len(net_files), figsize=(6*len(net_files), 6))
    if len(net_files) == 1:
        axes = [axes]

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

        C = Computer(chess.WHITE)

        hardcode_scores = []
        nnue_scores = []

        with torch.no_grad():
            for i in range(num_positions):
                board = random_board()

                # NNUE evaluation
                nnue_score = C.evaluate(board)
                nnue_normalised = C.normalise_score(nnue_score)
                nnue_scores.append(nnue_normalised)

                # Hardcode evaluation
                hardcode_score = C.hardcode_evaluate(board)
                hardcode_normalised = C.normalise_score(hardcode_score)
                hardcode_scores.append(hardcode_normalised)

        ax = axes[idx]
        ax.scatter(hardcode_scores, nnue_scores, alpha=0.5)
        ax.set_xlabel('Hardcode Evaluation')
        ax.set_ylabel('NNUE Prediction')
        ax.set_title(f'{net_file}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate scatter plots comparing hardcode evaluation to NNUE predictions for each net.")
    parser.add_argument('-n', '--num_positions', type=int, default=1000, help='Number of random positions to evaluate (default: 1000)')
    args = parser.parse_args()

    main(args.num_positions)
