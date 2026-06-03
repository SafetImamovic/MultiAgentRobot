import numpy as np
import json
from generator import generate_connected_room
from agent import Agent

boje: list[(int, int, int)] = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class State:
    def __init__(self, grid_size, num_agents):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.step_count = 0
        self.grid = None
        self.agents = []  # lista Agent objekata
        self.init_grid = None

    def reset(self):
        self.grid = generate_connected_room(self.grid_size)

        self.init_grid = self.grid.copy()

        self.agents = []

        for i in range(self.num_agents):
            while True:
                x = np.random.randint(1, self.grid_size[0] - 1)

                y = np.random.randint(1, self.grid_size[1] - 1)

                if (
                    all(agent.pos != (x, y) for agent in self.agents)
                    and self.grid[x, y] == 0
                ):
                    self.agents.append(
                        Agent(i, (x, y), self.grid_size, boja=boje[i % len(boje)])
                    )
                    break
        self.step_count = 0

        return self.get_state()

    def step(self, actions):
        rewards = []

        positions = [agent.pos for agent in self.agents]

        for agent, action in zip(self.agents, actions):
            # proslijedi ostale pozicije da agent izbjegne koliziju
            other_positions = [p for p in positions if p != agent.pos]

            reward = agent.akcija(action, self.grid, other_positions)

            rewards.append(reward)

        self.step_count += 1

        done = np.all(self.grid != 0)

        if done:
            for agent in self.agents:
                agent.last_reward += 200.0 / self.num_agents

            rewards = [agent.last_reward for agent in self.agents]

        return self.get_state(), rewards, done

    def get_state(self):
        return {
            "init_grid": self.init_grid.tolist(),
            "grid": self.grid.tolist(),
            "agents": [agent.get_data() for agent in self.agents],
            "step_count": self.step_count,
        }

    def save_to_json(self, filename="episode_log.json", mode="a"):
        """
        Zapiši trenutno stanje u JSON fajl.
        Ako želiš da svaki timestep bude niz u JSON fajlu, koristi "append mode"
        """
        state_dict = self.get_state()
        with open(filename, mode) as f:
            json.dump(state_dict, f)
            f.write("\n")  # svaka epizoda/timestep u novom redu
