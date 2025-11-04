import sys
sys.path.insert(0, 'chess_bot')

import chess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np

import random as rnd
from tqdm import tqdm

from bot.main import Computer

####################################################################################################
# TRAINING DATA AND CONSTANTS

LEARNING_RATE = 1e-3
EPOCHS = int(1e3)
BATCH_SIZE = 64

C = Computer(chess.WHITE)
evaluate = C.evaluate

np.random.RandomState(1)

####################################################################################################
# MODEL

from nnue.model import Net

net = Net()
criterion = nn.SmoothL1Loss()

try:
    net.load()
except FileNotFoundError:
    print("Could not find net, using random weights")
    net.random()
optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


####################################################################################################
# TRAINING

def random_board() -> chess.Board:
    board = chess.Board()
    for i in range(0, np.random.randint(0, 100)):
        legal_moves = list(board.legal_moves)
        move = rnd.choice(legal_moves)
        board.push(move)

        if board.is_game_over():
            break

    return board

def XY_pair() -> tuple[torch.Tensor, torch.Tensor]:
    board = random_board()

    # evaluate the board
    value = C.normalise_score(evaluate(board)) / 10

    # convert the board to a feature vector
    feat_vector = C.board_to_feat_vector(board)

    # return the feature vector and the value
    return feat_vector, torch.tensor([value], dtype=torch.float32)

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
    for i in pbar:
        try:
            feat_vector, value = XY_batch()
            optimiser.zero_grad()
            output = net(feat_vector)
            criterion(output, value).backward()
            optimiser.step()

            if i % (EPOCHS / 100) == 0:
                pbar.set_postfix(loss=f"{criterion(output, value).item():.4f}")
            if i % (EPOCHS / 10) == 0:
                sample(3)
        
        except (Exception, KeyboardInterrupt) as e:
            print("Error! Saving net...")
            net.save()
            raise e

    print("Saving net...")
    net.save()


def sample(n: int):
    """Sample the net by passing in a random position and comparing the returned evaluation."""

    for _ in range(n):
        board = random_board()

        # evaluate the board
        value = C.normalise_score(evaluate(board)) / 10

        # convert the board to a feature vector
        feat_vector = C.board_to_feat_vector(board)

        # return the feature vector and the value
        output = net(feat_vector)
        print(f"""T {value:.2f} | E {output.item():.2f} | e {round(output.item() - value, 2)}""")


while True:
    train(EPOCHS)