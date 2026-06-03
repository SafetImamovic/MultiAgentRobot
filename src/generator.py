import numpy as np
import random
from collections import deque
import config


def generate_connected_room(
    grid_size, wall_fraction=config.WALL_FRACTION, max_attempts=100
):
    """
    Generiše grid sa zidovima i praznim prostorom,
    tako da su sve slobodne ćelije povezane.
    """
    for attempt in range(max_attempts):
        grid = np.zeros(grid_size)

        # Dodaj granicne zidove
        # grid[0, :] = 1
        # grid[-1, :] = 1
        # grid[:, 0] = 1
        # grid[:, -1] = 1

        # Random zidovi unutar sobe
        for i in range(1, grid_size[0] - 1):
            for j in range(1, grid_size[1] - 1):
                if random.random() < wall_fraction:
                    grid[i, j] = 1

        # Provjera povezanosti
        if is_connected(grid):
            return grid

    # fallback ako nismo našli validnu sobu
    raise ValueError("Nije moguće generisati povezanu sobu")


def is_connected(grid):
    """
    Provjerava da li su sve slobodne ćelije povezane.
    BFS od prve slobodne ćelije.
    """
    free_cells = [
        (i, j)
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
        if grid[i, j] == 0
    ]
    if not free_cells:
        return False

    visited = set()
    queue = deque()
    queue.append(free_cells[0])
    visited.add(free_cells[0])

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return len(visited) == len(free_cells)
