from state import State
import config
import numpy as np
from typing import List
from config import Akcija


class SobaGridEnv:
    def __init__(
        self,
        broj_epizode: int = 0,
        grid_size=config.GRID_SIZE,
        num_agents=config.NUM_AGENTS,
    ):
        self.broj_epizode = broj_epizode
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.state = State(grid_size, num_agents)
        self.prvi_korak: bool = True

    def reset(self):
        return self.state.reset()

    def step(self, actions: List[Akcija]):
        rewards: List[float] = []

        current_positions = [agent.pos for agent in self.state.agents]

        for i, action in enumerate(actions):
            agent = self.state.agents[i]

            other_positions = [p for j, p in enumerate(current_positions) if j != i]

            reward = agent.akcija(action, self.state.grid, other_positions)

            rewards.append(reward)

            current_positions[i] = agent.pos

        self.state.step_count += 1

        done = np.all(self.state.grid != 0) or self.state.step_count >= config.MAX_STEP

        shared_reward = 0.0

        if done:
            if self.state.step_count < config.MAX_STEP:
                print("Nagrada!")
                shared_reward = config.REWARD_COMPLETE / self.num_agents

            for agent in self.state.agents:
                agent.last_reward += shared_reward

            rewards = [agent.last_reward for agent in self.state.agents]

            # self.state.save_to_json(
            #    filename=f"epizode/epizoda_{self.broj_epizode}_log.json"
            # )

        return self.state.get_state(), rewards, done
