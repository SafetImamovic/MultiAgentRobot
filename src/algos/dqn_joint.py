import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from roomba_env import MultiRoombaGrid
from env_wrappers import FlatActionWrapper
from algos.common import set_all_seeds, evaluate_policy, save_metrics
import numpy as np


def make_env(env_config, log_dir=None):
    env = MultiRoombaGrid(
        grid_size=env_config.get("grid_size", 8),
        n_roombas=env_config.get("n_roombas", 2),
        render_mode=None,
        max_steps=env_config.get("max_steps", 500),
        verbose=False,
        log_dir=log_dir,
    )
    return FlatActionWrapper(env)


def train(env_config, seed, total_timesteps, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)
    set_random_seed(seed)

    env = make_env(env_config, log_dir=log_dir)
    env.reset(seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.05,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        tau=0.005,
        train_freq=4,
        learning_starts=1000,
        verbose=1,
        device="cpu",
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model_path = os.path.join(log_dir, "dqn_joint_final")
    model.save(model_path)
    env.close()
    return model_path


def evaluate(model_path, env_config, seed, n_episodes, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)

    model = DQN.load(model_path)
    raw_env = MultiRoombaGrid(
        grid_size=env_config.get("grid_size", 8),
        n_roombas=env_config.get("n_roombas", 2),
        render_mode=None,
        max_steps=env_config.get("max_steps", 500),
        verbose=False,
        log_dir=None,
    )
    wrapped_env = FlatActionWrapper(raw_env)
    wrapped_env.reset(seed=seed)

    n_agents = env_config.get("n_roombas", 2)
    actions_per_agent = 5

    def policy_fn(obs):
        flat_action, _ = model.predict(obs, deterministic=True)
        # Decode flat action to multi-agent actions
        actions = []
        remaining = int(flat_action)
        for _ in range(n_agents):
            actions.append(remaining % actions_per_agent)
            remaining //= actions_per_agent
        return np.array(actions)

    metrics = evaluate_policy(policy_fn, raw_env, n_episodes)
    save_metrics(metrics, os.path.join(log_dir, "eval_metrics.json"))
    raw_env.close()
    return metrics
