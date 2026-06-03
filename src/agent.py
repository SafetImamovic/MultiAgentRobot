import numpy as np
import json
import config
from config import Akcija


class Agent:
    def __init__(self, agent_id, start_pos, grid_size, boja=(0, 0, 255)):
        self.id: int = agent_id

        self.pos: (int, int) = start_pos  # (x, y)

        self.coverage_map = np.zeros(grid_size, dtype=int)

        self.last_reward: float = 0.0

        self.boja: (int, int, int) = boja

        self.koraci = []

        self.koraci.append(
            {
                "pozicija_x": self.pos[0],
                "pozicija_y": self.pos[1],
                "coverage_map": self.coverage_map,
                # ... rest of your data ...
            }
        )

    def akcija(self, action: Akcija, grid, other_positions):
        x, y = self.pos
        nx, ny = x, y
        reward = 0.0

        if action == Akcija.GORE.value:
            nx -= 1
        elif action == Akcija.DOLE.value:
            nx += 1
        elif action == Akcija.LIJEVO.value:
            ny -= 1
        elif action == Akcija.DESNO.value:
            ny += 1
        # čekanje = 4 stays in place

        # FIRST: Check and fix boundaries BEFORE using nx, ny as indices
        if nx >= grid.shape[0]:
            nx = grid.shape[0] - 1
            reward += config.REWARD_ILLEGAL
        elif nx < 0:
            nx = 0
            reward += config.REWARD_ILLEGAL

        if ny >= grid.shape[1]:
            ny = grid.shape[1] - 1
            reward += config.REWARD_ILLEGAL
        elif ny < 0:
            ny = 0
            reward += config.REWARD_ILLEGAL

        # NOW it's safe to use nx, ny as indices
        if grid[nx, ny] == 1 or (nx, ny) in other_positions:
            reward += config.REWARD_ILLEGAL
            nx, ny = x, y

        elif grid[nx, ny] == 0:
            reward += config.REWARD_CLEAN

            grid[nx, ny] = 2

        elif grid[nx, ny] == 2:
            reward += config.REWARD_CLEANED

        if action == Akcija.CEKAJ.value:
            reward += config.REWARD_WAIT

        # Update position
        self.pos = (nx, ny)

        # Mark coverage
        self.coverage_map[nx, ny] = 1

        self.last_reward = reward
        self.koraci.append(
            {
                "akcija": action,
                "nagrada": reward,
                "pozicija_x": self.pos[0],
                "pozicija_y": self.pos[1],
                "coverage_map": self.coverage_map,
                # ... rest of your data ...
            }
        )

        return reward

    def get_data(self):
        return {"id": self.id, "boja": self.boja, "koraci": self.koraci}
