import torch
import torch.nn as nn
import chess
import numpy as np

class Net(nn.Module):

    INPUT_FEATURES = 768
    OUTPUT_FEATURES = 1

    NET_PATH = "chess_bot/nnue/net.pt"

    def __init__(self, hidden_sizes: list[int] = [256, 128, 8]):
        super(Net, self).__init__()

        # generate hidden layers
        self.hidden_layers = nn.ModuleList()
        in_feat = self.INPUT_FEATURES
        for i in range(len(hidden_sizes)):
            layer = nn.Linear(in_feat, hidden_sizes[i])
            self.hidden_layers.append(layer)
            self.hidden_layers.append(nn.ReLU())
            in_feat = hidden_sizes[i]

        # generate output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def board_to_feat_vector(self, board: chess.Board) -> torch.Tensor:
        feat_vector = np.zeros(768, dtype=np.float32)
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type  # 1-6
                color = piece.color  # chess.WHITE or chess.BLACK
                # Index calculation: (piece_type - 1) * 128 + (1 if color == chess.BLACK else 0) * 64 + square
                index = (piece_type - 1) * 128 + (1 if color == chess.BLACK else 0) * 64 + square
                feat_vector[index] = 1.0
        return torch.tensor(feat_vector, dtype=torch.float32)

    def save(self):
        torch.save(self.state_dict(), self.NET_PATH)

    def load(self):
        self.load_state_dict(torch.load(self.NET_PATH))

    def random(self):
        """Fill the network with random weights."""

        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)
