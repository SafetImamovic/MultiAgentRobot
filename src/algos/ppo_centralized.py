import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from roomba_env import MultiRoombaGrid
from algos.common import set_all_seeds, evaluate_policy, EpisodeLogger, save_metrics


def make_env(env_config, log_dir=None, render_mode=None):
    return MultiRoombaGrid(
        grid_size=env_config.get("grid_size", 8),
        n_roombas=env_config.get("n_roombas", 2),
        render_mode=render_mode,
        max_steps=env_config.get("max_steps", 500),
        verbose=False,
        log_dir=log_dir,
    )


def train(env_config, seed, total_timesteps, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)
    set_random_seed(seed)

    # MlpPolicy is faster on CPU; GPU only helps with CNN policies
    device = "cpu"

    train_env = make_env(env_config, log_dir=log_dir)
    train_env.reset(seed=seed)

    eval_dir = os.path.join(log_dir, "eval")
    eval_env = make_env(env_config, log_dir=eval_dir)
    eval_env.reset(seed=seed + 1000)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=device,
        seed=seed,
    )

    save_path = os.path.join(log_dir, "models")
    os.makedirs(save_path, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=5000, save_path=save_path, name_prefix="ppo", verbose=0
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best"),
        log_path=eval_dir,
        eval_freq=2500,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model_path = os.path.join(log_dir, "ppo_final")
    model.save(model_path)
    train_env.close()
    eval_env.close()
    return model_path


def evaluate(model_path, env_config, seed, n_episodes, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)

    model = PPO.load(model_path)
    env = make_env(env_config)
    env.reset(seed=seed)

    def policy_fn(obs):
        actions, _ = model.predict(obs, deterministic=True)
        return actions

    metrics = evaluate_policy(policy_fn, env, n_episodes)
    save_metrics(metrics, os.path.join(log_dir, "eval_metrics.json"))
    env.close()
    return metrics
