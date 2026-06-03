import os
import csv
import json
import random
import numpy as np
import torch
from collections import deque

# ---------------------
# Seeding
# ---------------------

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------
# Replay Buffer
# ---------------------

class ReplayBuffer:
    """Simple replay buffer for joint transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done):
        self.buffer.append((obs, actions, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------
# DQN Network
# ---------------------

class DQNNetwork(torch.nn.Module):
    """Shared DQN architecture for IDQN, QMIX, Joint-DQN: 256-256-128."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------
# Episode Logger
# ---------------------

class EpisodeLogger:
    """Logs episode metrics to CSV."""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "episode_data.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "coverage_pct", "steps",
                             "total_collisions", "terminated"])

    def log(self, episode, total_reward, coverage_pct, steps, total_collisions, terminated):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward, coverage_pct, steps,
                             total_collisions, terminated])


# ---------------------
# Evaluation
# ---------------------

def evaluate_policy(policy_fn, env, n_episodes=100):
    """
    Evaluate a policy function on an environment.

    Args:
        policy_fn: callable(obs) -> actions (numpy array)
        env: MultiRoombaGrid instance
        n_episodes: number of evaluation episodes

    Returns:
        dict with success_rate, avg_reward, avg_coverage, avg_collisions
    """
    successes = []
    rewards = []
    coverages = []
    collisions = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            actions = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(actions)
            ep_reward += reward
            done = terminated or truncated

        coverage_pct = env.coverage.sum() / (env.grid_size ** 2)
        successes.append(float(terminated))
        rewards.append(ep_reward)
        coverages.append(coverage_pct)
        collisions.append(env.total_collisions)

    return {
        "success_rate": float(np.mean(successes) * 100),
        "avg_reward": float(np.mean(rewards)),
        "avg_coverage": float(np.mean(coverages) * 100),
        "avg_collisions": float(np.mean(collisions)),
    }


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
