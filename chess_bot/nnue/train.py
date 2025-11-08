import sys
sys.path.insert(0, 'chess_bot')

import argparse

import chess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os

import random as rnd
from tqdm import tqdm

from bot.main import Computer

####################################################################################################
# TRAINING DATA AND CONSTANTS

LEARNING_RATE = 1e-4
EPOCHS = 10
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
optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

####################################################################################################
# TRAINING

def random_board() -> chess.Board:
    board = chess.Board()
    for i in range(0, np.random.randint(100, 300)):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = rnd.choice(legal_moves)
        board.push(move)

        if board.is_game_over():
            break

    return board

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

    while not board.is_game_over(claim_draw=True):
        # Get the current player
        current_computer = white_computer if board.turn == chess.WHITE else black_computer

        # Evaluate the current position with hardcode evaluation
        eval_score = hardcode_evaluate(board)
        positions.append(net.board_to_feat_vector(board))
        evaluations.append(C.nnue_normalise_score(eval_score))

        move = current_computer.best_move(board, time_per_move=10)  # Short timeout for training

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

def train(n: int, silent: bool=False):
    """
    Train the NNUE network by playing games and collecting data.
    """
    net.train()
    all_positions = []
    all_targets = []

    print("Collecting training data...")
    for _ in tqdm(range(n)):
        if silent:
            sys.stdout = open(os.devnull, 'w')
        positions, targets = play_game()
        if silent:
            sys.stdout = sys.__stdout__
        all_positions.extend(positions)
        all_targets.extend(targets)

    # Split into train and validation sets
    train_size = int(0.8 * len(all_positions))
    val_size = len(all_positions) - train_size
    train_positions, val_positions = all_positions[:train_size], all_positions[train_size:]
    train_targets, val_targets = all_targets[:train_size], all_targets[train_size:]

    # Create datasets and dataloaders
    train_dataset = data.TensorDataset(torch.stack(train_positions), torch.tensor(train_targets, dtype=torch.float32))
    val_dataset = data.TensorDataset(torch.stack(val_positions), torch.tensor(val_targets, dtype=torch.float32))
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} positions, validating on {len(val_dataset)} positions...")

    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training loop
        net.train()
        epoch_train_loss = 0.0
        for batch_positions, batch_targets in train_dataloader:
            optimiser.zero_grad()
            outputs = net(batch_positions)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimiser.step()
            epoch_train_loss += loss.item()

        # Validation loop
        net.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_positions, batch_targets in val_dataloader:
                outputs = net(batch_positions)
                loss = criterion(outputs.squeeze(), batch_targets)
                epoch_val_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        if len(val_dataloader) > 0:
            avg_val_loss = epoch_val_loss / len(val_dataloader)
        else:
            avg_val_loss = float('inf')
        scheduler.step(avg_val_loss)  # Update learning rate based on validation loss
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            net.save("chess_bot/nnue/nets/best_net.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NNUE model.")
    parser.add_argument('-n', '--name', help='Load the net from the specified *name*.')
    parser.add_argument('-s', '--silent', action='store_true', help='Specify to disable information regarding the gameplay, including board positions.')
    args = parser.parse_args()

    if args.name is None:
        print("Please specify a name for the net to load.")
        exit(1)

    # Load the net based on the argument
    try:
        path = "chess_bot/nnue/nets/" + args.name + ".pt"
        net.load(path)
        print(f"Loaded net from {args.name}.pt successfully")
    except FileNotFoundError:
        print("Could not find net, using random weights")
        net.random()

    while True:
        try:
            train(EPOCHS, silent=args.silent)
        except (KeyboardInterrupt, Exception) as e:
            print("Error! Saving net...")
            net.save("chess_bot/nnue/nets/net.pt")

            if type(e) == KeyboardInterrupt:
                raise e
            raise e
        else:
            net.save("chess_bot/nnue/nets/net.pt")
            print("Model saved.")
