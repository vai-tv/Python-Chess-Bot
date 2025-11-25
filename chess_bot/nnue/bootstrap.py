import sys
sys.path.insert(0, 'chess_bot')

import chess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import numpy as np

import random as rnd
from tqdm import tqdm

from bot.main import Computer

####################################################################################################
# TRAINING DATA AND CONSTANTS

LEARNING_RATE = 5e-3
EPOCHS = int(1e3)
BATCH_SIZE = 128

C = Computer(chess.WHITE)
evaluate = C.hardcode_evaluate

np.random.RandomState(1)

####################################################################################################
# MODEL

from nnue.model import Net

net = Net()
net_path = "chess_bot/nnue/nets/bootstrap.pt"
criterion = nn.SmoothL1Loss()

try:
    net.load(net_path)
except FileNotFoundError:
    print("Could not find net, using random weights")
    net.random()
optimiser = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.8, patience=100)


####################################################################################################
# TRAINING

def random_board() -> chess.Board:
    board = chess.Board()
    for i in range(0, np.random.poisson(25, size=2)[0]):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = rnd.choice(legal_moves)
        board.push(move)

        if board.is_game_over():
            break

    return board

def XY_pair() -> tuple[torch.Tensor, torch.Tensor]:
    board = random_board()

    # evaluate the board
    value = np.tanh(C.nnue_normalise_score(evaluate(board)))

    # convert the board to a feature vector
    feat_vector = net.board_to_feat_vector(board)

    # return the feature vector and the value
    return feat_vector, torch.tensor(value, dtype=torch.float32)

def XY_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Similar to XY_pair, but returns a batch of feature vectors and values."""

    feat_vectors = []
    values = []

    for _ in range(BATCH_SIZE):
        feat_vector, value = XY_pair()
        feat_vectors.append(feat_vector)
        values.append(value)
    return torch.stack(feat_vectors), torch.stack(values)


def train(n: int):
    pbar = tqdm(range(n), desc="Training NNUE")
    for _ in pbar:
        try:
            feat_vector, value = XY_batch()
            optimiser.zero_grad()
            output = net(feat_vector)
            loss = criterion(output, value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimiser.step()
            scheduler.step(loss.detach())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimiser.param_groups[0]['lr']:.6f}")       

        except (Exception, KeyboardInterrupt) as e:
            print("Error! Saving net...")
            net.save(net_path)
            raise e

    print("Saving net...")
    net.save(net_path)
    sample(3)


def sample(n: int):
    """Sample the net by passing in a random position and comparing the returned evaluation."""

    print()
    for _ in range(n):
        board = random_board()

        # evaluate the board
        value = C.nnue_normalise_score(evaluate(board))

        # convert the board to a feature vector
        feat_vector = net.board_to_feat_vector(board)

        # return the feature vector and the value
        output = np.arctanh(net(feat_vector).item())
        print(f"""T {value:.2f} | E {output:.2f} | e {round(output - value, 2)}""")
        # print(board)


if __name__ == '__main__':
    while True:
        train(EPOCHS)
