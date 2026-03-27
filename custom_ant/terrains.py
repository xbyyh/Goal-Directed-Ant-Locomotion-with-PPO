# custom_ant/terrains.py

import numpy as np


def flat(nrow: int, ncol: int) -> np.ndarray:
    """Flat terrain at height 0."""
    return np.zeros((nrow, ncol), dtype=np.float32)


def slope(nrow: int, ncol: int) -> np.ndarray:
    """
    Uphill slope terrain covering the entire heightfield.


    Based on testing: MuJoCo heightfield has:
      - row index increases in y direction
      - col index increases in x direction

    """
    # Create height that increases with row and col indices
    
    row_grad = np.linspace(0.25, 0, nrow, dtype=np.float32)
    col_grad = np.linspace(0.25, 0, ncol, dtype=np.float32)

    row_component = np.tile(row_grad.reshape(-1, 1), (1, ncol))
    col_component = np.tile(col_grad.reshape(1, -1), (nrow, 1))

    # Average for smooth diagonal slope: 0 at [0,0], 1 at [nrow-1,ncol-1]
    height = 0.5 * (row_component + col_component)

    return height.astype(np.float32)


def bumpy(nrow: int, ncol: int) -> np.ndarray:
    """
    Bumpy terrain like a sine wave going up and down.
    Creates smooth oscillations across the terrain.
    """
    xs = np.linspace(0.0, 6 * np.pi, ncol, dtype=np.float32)
    ys = np.linspace(0.0, 6 * np.pi, nrow, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # Sine wave pattern - bumps going up and down
    height = 0.15 * np.sin(X) * np.sin(Y)
    return height.astype(np.float32)

def rumble_strips(nrow: int, ncol: int) -> np.ndarray:
    x = np.linspace(100.0, 2 * np.pi * 75, ncol, dtype=np.float32)
    height = 0.04 * np.sin(x)          # shallow periodic ridges
    return np.tile(height, (nrow, 1))


TERRAIN_MAP = {
    "flat": flat,
    "slope": slope,
    "bumpy": bumpy,
    "rumble": rumble_strips,
}
