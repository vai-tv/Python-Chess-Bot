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
EPOCHS = int(1e1)
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor for gamma weighting

C = Computer(chess.WHITE)
nnue_evaluate = C.evaluate
hardcode_evaluate = C.hardcode_evaluate

np.random.RandomState(1)

####################################################################################################
# MODEL

from nnue.model import Net

net = Net()
criterion = nn.SmoothL1Loss()

try:
    net.load("chess_bot/nnue/bootstrap.pt")
except FileNotFoundError:
    print("Could not find net, using random weights")
    net.random()
optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

####################################################################################################
# TRAINING

def random_board() -> chess.Board:
    board = chess.Board()
    for i in range(0, np.random.randint(100, 200)):
        legal_moves = list(board.legal_moves)
        move = rnd.choice(legal_moves)
        board.push(move)

        if board.is_game_over():
            break

    return board


def sample(n: int):
    """Sample the net by passing in a random position and comparing the returned evaluation."""

    for _ in range(n):
        board = random_board()

        # evaluate the board
        value = C.normalise_score(hardcode_evaluate(board)) / 10

        # convert the board to a feature vector
        feat_vector = net.board_to_feat_vector(board)

        # return the feature vector and the value
        output = net(feat_vector)
        print(f"""T {value:.2f} | E {output.item():.2f} | e {round(output.item() - value, 2)}""")


def play_game() -> tuple[list[torch.Tensor], list[float]]:
    """
    Play a game between two instances of the computer using NNUE evaluation.
    Collect all positions and their evaluations.
    """
    board = chess.Board()
    positions = []
    evaluations = []

    # Create two computer instances, one for white, one for black
    white_computer = Computer(chess.WHITE)
    black_computer = Computer(chess.BLACK)

    while not board.is_game_over():
        # Get the current player
        current_computer = white_computer if board.turn == chess.WHITE else black_computer

        # Evaluate the current position with NNUE
        eval_score = nnue_evaluate(board)
        positions.append(net.board_to_feat_vector(board))
        evaluations.append(eval_score)

        move = current_computer.best_move(board, time_per_move=7.5)  # Short timeout for training

        if move is None:
            break

        print(board.san(move))
        board.push(move)
        print(board)

    # Determine the game result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = -1.0  # Black wins
        else:
            result = 1.0   # White wins
    else:
        result = 0.0  # Draw

    # Apply gamma weighting to targets
    targets = []
    for i in range(len(evaluations)):
        # Weight the evaluation towards the game result
        # Closer positions to the end have higher weight on the result
        weight = GAMMA ** (len(evaluations) - 1 - i)
        target = (1 - weight) * evaluations[i] + weight * result
        targets.append(target)

    return positions, targets

def train(n: int):
    """
    Train the NNUE network by playing games and collecting data.
    """
    net.train()
    all_positions = []
    all_targets = []

    print("Collecting training data...")
    for _ in tqdm(range(n)):
        positions, targets = play_game()
        all_positions.extend(positions)
        all_targets.extend(targets)

    # Create dataset and dataloader
    dataset = data.TensorDataset(torch.stack(all_positions), torch.tensor(all_targets, dtype=torch.float32))
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training on {len(dataset)} positions...")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_positions, batch_targets in dataloader:
            optimiser.zero_grad()
            outputs = net(batch_positions)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save the trained model
    net.save()
    print("Model saved.")


if __name__ == "__main__":
    while True:
        train(EPOCHS)
