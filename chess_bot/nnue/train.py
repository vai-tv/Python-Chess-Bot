import sys
sys.path.insert(0, 'chess_bot')

import argparse
import chess
import json
import multiprocessing
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm

from bot.main import Computer

####################################################################################################
# TRAINING DATA AND CONSTANTS

LEARNING_RATE = 1e-4  # Increased learning rate for faster convergence
EPOCHS = 25
BATCH_SIZE = 128
GAMMA = 0.95

C = Computer(chess.WHITE)
nnue_evaluate = C.evaluate
hardcode_evaluate = C.hardcode_evaluate

np.random.RandomState(1)

workers = multiprocessing.cpu_count()

####################################################################################################
# MODEL

from nnue.model import Net

net = Net()
# Changed loss: combined MSELoss and SmoothL1Loss
mse_criterion = nn.MSELoss()
smoothl1_criterion = nn.SmoothL1Loss()
optimiser = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

####################################################################################################
# TRAINING

OPENING_BOOK_PATH = "chess_bot/play/openings.json"

def random_start_board() -> chess.Board:
    """Select a random starting position from the opening book."""

    with open(OPENING_BOOK_PATH, "r") as f:
        openings = json.load(f)

    random_opening = np.random.choice(openings["15"])
    
    return chess.Board(random_opening)

def augment_feat_vector(feat_vector: torch.Tensor, target: float) -> list[tuple[torch.Tensor, float]]:
    """Generate augmented feature vectors and targets by applying both augmentations."""
    augmented = [(feat_vector, target)]  # Original

    # Horizontal flip: swap files a-h, b-g, etc., ignoring last 13 special bits
    flipped = feat_vector.clone()
    # Pieces: 768 features, 6 types * 2 colors * 64 squares
    for piece_type in range(6):
        for color in range(2):
            for rank in range(8):
                # Collect the values before swapping to avoid in-place overwrite issues
                row = [flipped[piece_type * 128 + color * 64 + rank * 8 + file] for file in range(8)]
                # Swap files in row
                for file in range(8):
                    new_file = 7 - file
                    flipped[piece_type * 128 + color * 64 + rank * 8 + file] = row[new_file]
    augmented.append((flipped, target))

    # Color swap: swap white and black pieces, reverse the target
    swapped = feat_vector.clone()
    # Swap white and black for each piece type and square
    for piece_type in range(6):
        for rank in range(8):
            # Collect values before swapping to avoid overwrite issues
            white_row = [swapped[piece_type * 128 + 0 * 64 + rank * 8 + file] for file in range(8)]
            black_row = [swapped[piece_type * 128 + 1 * 64 + rank * 8 + file] for file in range(8)]
            for file in range(8):
                swapped[piece_type * 128 + 0 * 64 + rank * 8 + file] = black_row[file]
                swapped[piece_type * 128 + 1 * 64 + rank * 8 + file] = white_row[file]
    augmented.append((swapped, -target))

    return augmented

def play_game(silent: bool = False) -> tuple[list[torch.Tensor], list[float]]:
    """
    Play a game between two instances of the computer using NNUE evaluation.
    Collect all positions and their evaluations.
    """
    board = random_start_board()
    if not silent:
        print("Starting position:", board.fen())

    positions = []
    evaluations = []

    # Create two computer instances, one for white, one for black
    # Set workers=1 to avoid multiprocessing issues in daemonic processes
    white_computer = Computer(chess.WHITE, workers=1)
    black_computer = Computer(chess.BLACK, workers=1)

    if not silent:
        print(board)

    while not board.is_game_over(claim_draw=True):
        # Get the current player
        current_computer = white_computer if board.turn == chess.WHITE else black_computer

        # Evaluate the current position with hardcode evaluation
        eval_score = hardcode_evaluate(board)
        feat_vector = net.board_to_feat_vector(board)
        positions.append(feat_vector)
        evaluations.append(np.tanh(C.nnue_normalise_score(eval_score)))

        # Variable move time
        ply = board.ply()
        if ply < 20:
            time_per_move = 5
        else:
            time_per_move = 7.5

        if silent:
            sys.stdout = open(os.devnull, 'w')
        move = current_computer.best_move(board, time_per_move=time_per_move)
        if silent:
            sys.stdout = sys.__stdout__

        if move is None:
            break

        if not silent:
            print(board.san(move))
        board.push(move)
        if not silent:
            print(board)

    n = len(evaluations)

    # Determine the game result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = -1  # Black wins
        else:
            result = 1   # White wins
    else:
        result = 0.0  # Draw

    # Apply gamma weighting to targets
    targets = [0] * n
    next_target = result  # +1 / 0 / -1

    for i in reversed(range(n)):
        v = evaluations[i]
        # one-step TD target
        target = v + GAMMA * (next_target - v)
        targets[i] = target
        next_target = target

    # Augment data
    # augmented_positions = []
    # augmented_targets = []
    # for pos, targ in zip(positions, targets):
    #     augmented = augment_feat_vector(pos, targ)
    #     for aug_pos, aug_targ in augmented:
    #         augmented_positions.append(aug_pos)
    #         augmented_targets.append(aug_targ)

    # return augmented_positions, augmented_targets

    return positions, targets #type: ignore

def collect_data(n: int, silent: bool = False) -> tuple[list[torch.Tensor], list[float]]:
    """
    Collect training data by playing games in parallel.
    """
    print("Collecting training data...")

    if silent:
        sys.stdout = open(os.devnull, 'w')

    try:
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap(play_game, [silent] * n), total=n))
    except Exception as e:
        print(f"Error collecting data. {e}")
        return [], []

    if silent:
        sys.stdout = sys.__stdout__

    all_positions = []
    all_targets = []
    for positions, targets in results:
        all_positions.extend(positions)
        all_targets.extend(targets)
    return all_positions, all_targets

def train_model(all_positions: list[torch.Tensor], all_targets: list[float], silent: bool = False):
    """
    Train the NNUE network on the provided data.
    """
    if not all_positions:
        print("No data to train on.")
        return

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(all_positions))
    val_size = int(0.2 * len(all_positions))
    test_size = len(all_positions) - train_size - val_size
    train_positions, val_positions, test_positions = all_positions[:train_size], all_positions[train_size:train_size+val_size], all_positions[train_size+val_size:]
    train_targets, val_targets, test_targets = all_targets[:train_size], all_targets[train_size:train_size+val_size], all_targets[train_size+val_size:]

    # Create datasets and dataloaders
    train_dataset = data.TensorDataset(torch.stack(train_positions), torch.tensor(train_targets, dtype=torch.float32))
    val_dataset = data.TensorDataset(torch.stack(val_positions), torch.tensor(val_targets, dtype=torch.float32))
    test_dataset = data.TensorDataset(torch.stack(test_positions), torch.tensor(test_targets, dtype=torch.float32))
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} positions, validating on {len(val_dataset)} positions, testing on {len(test_dataset)} positions...")

    # Reset learning rate and scheduler for each training session
    optimiser.param_groups[0]['lr'] = LEARNING_RATE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    patience = 20  # Increased patience for early stopping to allow longer training
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training loop
        net.train()
        epoch_train_loss = 0.0
        for batch_positions, batch_targets in train_dataloader:
            optimiser.zero_grad()
            outputs = net(batch_positions)
            # Combine MSE and SmoothL1 losses
            mse_loss = mse_criterion(outputs, batch_targets)
            smoothl1_loss = smoothl1_criterion(outputs, batch_targets)
            loss = 0.5 * mse_loss + 0.5 * smoothl1_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_train_loss += loss.item()

        # Validation loop
        net.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_positions, batch_targets in val_dataloader:
                outputs = net(batch_positions)
                mse_loss = mse_criterion(outputs, batch_targets)
                smoothl1_loss = smoothl1_criterion(outputs, batch_targets)
                loss = 0.5 * mse_loss + 0.5 * smoothl1_loss
                epoch_val_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        if len(val_dataloader) > 0:
            avg_val_loss = epoch_val_loss / len(val_dataloader)
        else:
            avg_val_loss = float('inf')
        scheduler.step(epoch)
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

    # Final test evaluation
    try:
        net.load("chess_bot/nnue/nets/best_net.pt")
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_positions, batch_targets in test_dataloader:
                outputs = net(batch_positions)
                # Use combined mse and smoothl1 loss here as well
                mse_loss = mse_criterion(outputs, batch_targets)
                smoothl1_loss = smoothl1_criterion(outputs, batch_targets)
                loss = 0.5 * mse_loss + 0.5 * smoothl1_loss
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Final Test Loss: {avg_test_loss:.4f}")
    except FileNotFoundError:
        print("No best model saved, skipping final test evaluation.")
    except Exception as e:
        print(f"Exception during final test evaluation: {e}")

def train(num_games: int, silent: bool=False):
    """
    Train the NNUE network by playing games and collecting data.
    """
    all_positions, all_targets = collect_data(num_games, silent)
    train_model(all_positions, all_targets, silent)
    


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
        all_positions = []
        all_targets = []
        try:
            all_positions, all_targets = collect_data(EPOCHS, silent=args.silent)
            train_model(all_positions, all_targets, silent=args.silent)
        except (KeyboardInterrupt) as e:
            print("Keyboard interrupt, training on collected data before exit...")
            if all_positions and all_targets:
                train_model(all_positions, all_targets, silent=args.silent)
            print("Saving net...")
            net.save("chess_bot/nnue/nets/net.pt")

            if type(e) == KeyboardInterrupt:
                raise e
        else:
            net.save("chess_bot/nnue/nets/net.pt")
            print("Model saved.")
