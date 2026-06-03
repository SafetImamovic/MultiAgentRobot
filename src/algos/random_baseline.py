import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roomba_env import MultiRoombaGrid
from algos.common import set_all_seeds, evaluate_policy, save_metrics


def train(env_config, seed, total_timesteps, log_dir):
    """Random baseline has no training. Returns None as model path."""
    os.makedirs(log_dir, exist_ok=True)
    return None


def evaluate(model_path, env_config, seed, n_episodes, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)

    env = MultiRoombaGrid(
        grid_size=env_config.get("grid_size", 8),
        n_roombas=env_config.get("n_roombas", 2),
        render_mode=None,
        max_steps=env_config.get("max_steps", 500),
        verbose=False,
        log_dir=None,
    )
    env.reset(seed=seed)

    def policy_fn(obs):
        return env.action_space.sample()

    metrics = evaluate_policy(policy_fn, env, n_episodes)
    save_metrics(metrics, os.path.join(log_dir, "eval_metrics.json"))
    env.close()
    return metrics
