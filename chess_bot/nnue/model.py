import torch
import torch.nn as nn
import chess
import numpy as np

class Net(nn.Module):

    INPUT_FEATURES = 768 + 4 + 8 + 1  # Pieces + castling + en passant + side to move
    OUTPUT_FEATURES = 1

    def __init__(self, hidden_sizes: list[int] = [128, 64, 32, 16], dropout_rate: float = 0.5):
        super(Net, self).__init__()

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residuals = nn.ModuleList()

        in_feat = self.INPUT_FEATURES
        for size in hidden_sizes:
            self.linears.append(nn.Linear(in_feat, size))
            self.norms.append(nn.LayerNorm(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            self.residuals.append(nn.Linear(in_feat, size) if in_feat != size else nn.Identity())
            in_feat = size

        # generate output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        for linear, norm, dropout, residual in zip(self.linears, self.norms, self.dropouts, self.residuals):
            out = linear(x)
            out = norm(out)
            out = nn.SiLU()(out)
            out = dropout(out)
            res = residual(x)
            x = out + res
        x = self.output_layer(x)
        return x.squeeze(-1)

    def board_to_feat_vector(self, board: chess.Board) -> torch.Tensor:
        feat_vector = np.zeros(self.INPUT_FEATURES, dtype=np.float32)
        offset = 0

        # Pieces on board (768 features)
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type  # 1-6
                color = piece.color  # chess.WHITE or chess.BLACK
                # Index calculation: (piece_type - 1) * 128 + (1 if color == chess.BLACK else 0) * 64 + square
                index = (piece_type - 1) * 128 + (1 if color == chess.BLACK else 0) * 64 + square
                feat_vector[index] = 1.0
        offset += 768

        # Castling rights (4 features)
        feat_vector[offset + 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        feat_vector[offset + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        feat_vector[offset + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        feat_vector[offset + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        offset += 4

        # En passant (8 features, one for each file a-h)
        if board.ep_square is not None:
            file = chess.square_file(board.ep_square)
            feat_vector[offset + file] = 1.0
        offset += 8

        # Side to move (1 feature)
        feat_vector[offset] = 1.0 if board.turn == chess.WHITE else 0.0

        return torch.tensor(feat_vector, dtype=torch.float32)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def random(self):
        """Fill the network with random weights."""

        for linear in self.linears:
            torch.nn.init.xavier_uniform_(linear.weight)    #type: ignore
            torch.nn.init.zeros_(linear.bias)               #type: ignore

        for residual in self.residuals:
            if isinstance(residual, nn.Linear):
                torch.nn.init.xavier_uniform_(residual.weight)  # type: ignore
                torch.nn.init.zeros_(residual.bias)  # type: ignore

        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.zeros_(self.output_layer.bias)

if __name__ == '__main__':
    net = Net()
    net.eval()
    net.random()
    net.forward(net.board_to_feat_vector(chess.Board()))
