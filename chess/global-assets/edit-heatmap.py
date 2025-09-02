import tkinter as tk
from tkinter import messagebox
import numpy as np
from matplotlib import colormaps as cm
import matplotlib.colors as colors
import json

HEATMAP = {
    "early": {
        "P": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0], [1.0, 1.5, 2.0, 3.5, 3.5, 2.0, 1.5, 1.0], [0.0, -0.5, -0.25, 3.0, 3.0, -0.25, -0.5, 0.0], [0.0, -0.5, -1.0, -0.5, -0.5, -1.0, -0.5, 0.0], [1.0, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "N": [[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0], [-4.0, -2.0, -0.5, -0.5, -0.5, -0.5, -2.0, -4.0], [-3.0, 0.0, 0.5, 0.75, 0.75, 0.5, 0.0, -3.0], [-3.0, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -3.0], [-3.0, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -3.0], [-3.0, 0.0, 0.5, 0.75, 0.75, 0.5, 0.0, -3.0], [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0], [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]],
        "B": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [-1.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, -1.0], [-1.0, 0.125, 0.5, 0.5, 0.5, 0.5, 0.125, -1.0], [-1.0, 0.25, 0.875, 0.5, 0.5, 0.875, 0.25, -1.0], [-1.0, 0.375, 0.625, 0.75, 0.75, 0.625, 0.375, -1.0], [-1.5, 0.5, 0.125, 0.25, 0.25, 0.125, 0.5, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
        "R": [[-0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.25], [0.5, 1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.75, 0.25, 0.375, 0.75, 0.75, 0.375, 0.25, -0.5], [-1.0, -0.75, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0]],
        "Q": [[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]],
        "K": [[-4, -5, -5, -5, -5, -5, -5, -4], [-4, -5, -5, -5, -5, -5, -5, -4], [-3, -5, -5, -5, -5, -5, -5, -3], [-3, -4, -5, -5, -5, -5, -4, -3], [-2, -3, -4, -5, -5, -4, -3, -2], [-1, -2, -3, -4, -4, -3, -2, -1], [0, -1, -2, -3, -3, -2, -1, 0], [2, 3, 2, 0, 0, 2, 4, 2]]
    },
    "middle": {
        "P": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.0], [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0], [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0], [0.5, 0.0, -0.5, 2.0, 2.0, -0.5, 0.0, 0.5], [0.5, 0.25, -1.0, 1.0, 1.0, -1.0, 0.25, 0.5], [1.0, 1.0, 1.0, -1.5, -1.5, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "N": [[-4.0, -3.0, -2.0, -2.0, -2.0, -2.0, -3.0, -4.0], [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0], [-2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, -2.0], [-2.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, -2.0], [-2.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, -2.0], [-2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, -2.0], [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0], [-4.0, -3.0, -2.0, -2.0, -2.0, -2.0, -3.0, -4.0]],
        "B": [[-1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -1.0, -1.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -0.5], [-0.5, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -0.5], [-0.5, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -0.5], [-0.5, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, -0.5], [-1.0, 0.625, 0.5, 0.5, 0.5, 0.5, 0.625, -1.0], [-1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -1.0, -1.5]],
        "R": [[-0.25, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, -0.25], [0.5, 1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.5, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, -0.5], [-0.75, 0.25, 0.375, 0.75, 0.75, 0.375, 0.25, -0.5], [-1.0, -0.75, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0]],
        "Q": [[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-0.5, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -0.5], [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]],
        "K": [[-10.0, -9.0, -8.0, -8.0, -8.0, -8.0, -9.0, -10.0], [-7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0], [-5.0, -5.0, -6.0, -6.0, -6.0, -6.0, -5.0, -5.0], [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0], [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0], [0.5, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.5], [1.0, 2.0, 3.0, -1.0, -1.0, 0.0, 3.0, 1.0]]
    },
    "late": {
        "P": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.5, -1.5, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        "N": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, -0.875, 0.0, 1.0, 1.0, 0.0, -0.875, -1.5], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, -1.0], [-1.0, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.5, -0.875, 0.0, 0.5, 0.5, 0.0, -0.875, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
        "B": [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]],
        "R": [[-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [1.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -0.5], [-0.5, 0.0, 1.0, 1.25, 1.25, 1.0, 0.0, -0.5], [-0.5, 0.0, 0.75, 1.0, 1.0, 0.75, 0.0, -0.5], [-1.0, -0.5, 0.5, 0.75, 0.75, 0.5, -0.5, -1.0]],
        "Q": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0], [-1.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, -1.5], [-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0]],
        "K": [[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0, -1.5, -2.0], [-1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1.5], [-1.0, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -1.0], [-1.0, -0.5, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0], [-1.0, -0.5, 0.5, 1.5, 1.5, 0.5, -0.5, -1.0], [-1.0, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -1.0], [-1.5, -1.25, -1.0, -0.75, -0.75, -1.0, -1.25, -1.5], [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]]
    }
}

target = HEATMAP['late']['K']

class HeatmapEditor:
    def __init__(self, master, initial_heatmap=None):
        self.master = master
        self.master.title("Heatmap Editor")
        self.grid_size = 8
        self.entries = []
        self.heatmap = np.zeros((self.grid_size, self.grid_size)) if initial_heatmap is None else np.array(initial_heatmap)
        self.cmap = cm['viridis']
        self.norm = colors.Normalize()  # Normalize heat values between min and max dynamically

        self.create_widgets()
        self.update_colors()

    def create_widgets(self):
        frame = tk.Frame(self.master)
        frame.pack(padx=10, pady=10)

        for row in range(self.grid_size):
            row_entries = []
            for col in range(self.grid_size):
                var = tk.StringVar()
                var.set(str(self.heatmap[row, col]))
                entry = tk.Entry(frame, width=5, justify='center', textvariable=var)
                entry.grid(row=row, column=col, padx=1, pady=1)
                entry.bind('<KeyRelease>', lambda event, r=row, c=col, v=var: self.on_value_change(r, c, v))
                entry.bind('<KeyPress-Up>', lambda event, r=row, c=col: self.on_arrow_key(event, r, c))
                entry.bind('<KeyPress-Down>', lambda event, r=row, c=col: self.on_arrow_key(event, r, c))
                entry.bind('<KeyPress-Left>', lambda event, r=row, c=col: self.on_arrow_key(event, r, c))
                entry.bind('<KeyPress-Right>', lambda event, r=row, c=col: self.on_arrow_key(event, r, c))
                row_entries.append((entry, var))
            self.entries.append(row_entries)

        btn_frame = tk.Frame(self.master)
        btn_frame.pack(pady=5)
        save_btn = tk.Button(btn_frame, text="Save Heatmap", command=self.save_heatmap)
        save_btn.pack()

    def on_value_change(self, row, col, var):
        val = var.get()
        try:
            fval = float(val)
            self.heatmap[row, col] = fval
            self.update_colors()  # Update entire grid colors on any value change
        except ValueError:
            # Invalid input, ignore color update
            pass

    def on_arrow_key(self, event, row, col):
        # Handle arrow key navigation between entries only if cursor at start or end
        entry_widget = self.entries[row][col][0]
        cursor_pos = entry_widget.index(tk.INSERT)
        text_len = len(entry_widget.get())

        if event.keysym == 'Up':
            new_row = (row - 1) % self.grid_size
            self.entries[new_row][col][0].focus_set()
        elif event.keysym == 'Down':
            new_row = (row + 1) % self.grid_size
            self.entries[new_row][col][0].focus_set()
        elif event.keysym == 'Left':
            if cursor_pos == 0:
                new_col = (col - 1) % self.grid_size
                self.entries[row][new_col][0].focus_set()
        elif event.keysym == 'Right':
            if cursor_pos == text_len:
                new_col = (col + 1) % self.grid_size
                self.entries[row][new_col][0].focus_set()

    def update_cell_color(self, row, col):
        # Normalize heatmap values dynamically between min and max
        min_val = np.min(self.heatmap)
        max_val = np.max(self.heatmap)
        val = self.heatmap[row, col]
        if max_val > min_val:
            norm_val = (val - min_val) / (max_val - min_val)
        else:
            norm_val = 0.0
        rgba = self.cmap(norm_val)
        # Convert rgba (numpy array) to tuple for rgb2hex
        rgba_tuple = tuple(rgba[:3])
        hex_color = colors.rgb2hex(rgba_tuple)
        entry, _ = self.entries[row][col]
        entry.config(bg=hex_color)

    def update_colors(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.update_cell_color(row, col)

    def save_heatmap(self):
        # Return heatmap as JSON string
        heatmap_list = self.heatmap.tolist()
        heatmap_str = json.dumps(heatmap_list)
        messagebox.showinfo("Heatmap Saved", "Heatmap saved as string:\n" + heatmap_str)
        self.master.quit()
        self.master.destroy()
        print(heatmap_str)  # Also print to stdout for capture if needed

def edit_heatmap(initial_heatmap=None):
    root = tk.Tk()
    editor = HeatmapEditor(root, initial_heatmap)
    root.mainloop()
    # After window closes, return the heatmap string
    return json.dumps(editor.heatmap.tolist())

if __name__ == "__main__":
    # Example usage with empty heatmap
    heatmap_str = edit_heatmap(target)
    print("Final heatmap string:")
    print(heatmap_str)
