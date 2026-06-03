import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FlatActionWrapper(gym.ActionWrapper):
    """Converts MultiDiscrete([5]*N) to Discrete(5^N) for SB3 DQN compatibility."""

    def __init__(self, env):
        super().__init__(env)
        self.n_agents = len(env.action_space.nvec)
        self.actions_per_agent = int(env.action_space.nvec[0])
        self.action_space = spaces.Discrete(self.actions_per_agent ** self.n_agents)

    def action(self, flat_action):
        actions = []
        remaining = int(flat_action)
        for _ in range(self.n_agents):
            actions.append(remaining % self.actions_per_agent)
            remaining //= self.actions_per_agent
        return np.array(actions)
