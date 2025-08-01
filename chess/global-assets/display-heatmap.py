import numpy as np
import matplotlib.pyplot as plt
import json
import os

heatmap = json.load(open(os.path.join(os.path.dirname(__file__), 'heatmap.json')))

def display_heatmap() -> None:
    """
    Display heatmaps for each game stage and piece type using matplotlib.pyplot.

    The heatmaps are displayed in a grid layout with titles and colorbars.
    """

    stages = list(heatmap.keys())  # e.g. ['early', 'middle', 'late']
    piece_symbols = list(next(iter(heatmap.values())).keys())  # e.g. ['P', 'N', 'B', 'R', 'Q', 'K']

    num_stages = len(stages)
    num_pieces = len(piece_symbols)

    fig, axes = plt.subplots(num_stages, num_pieces, figsize=(4 * num_pieces, 3 * num_stages), squeeze=False)

    for i, stage in enumerate(stages):
        for j, piece in enumerate(piece_symbols):
            ax = axes[i][j]
            data = np.array(heatmap[stage][piece])

            # Flip vertically to have rank 1 at bottom
            data = np.flipud(data)

            im = ax.imshow(data, cmap='hot', interpolation='nearest', origin='lower')

            ax.set_title(f"{stage.capitalize()} - {piece}")
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
            ax.set_yticklabels(range(1, 9))
            ax.grid(False)

            # Add colorbar to each subplot
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

display_heatmap()