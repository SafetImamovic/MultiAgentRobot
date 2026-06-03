import os
import sys
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roomba_env import MultiRoombaGrid
from algos.common import (
    set_all_seeds, DQNNetwork, ReplayBuffer,
    evaluate_policy, EpisodeLogger, save_metrics,
)


class IDQNTrainer:
    """Independent DQN: each agent has its own Q-network, trained independently."""

    def __init__(self, env_config, seed, n_agents=2, lr=1e-4, buffer_size=100_000,
                 batch_size=64, gamma=0.99, tau=0.005, eps_start=1.0, eps_end=0.05,
                 eps_decay_steps=500_000, warmup_steps=1000, train_freq=4):
        self.env = MultiRoombaGrid(
            grid_size=env_config.get("grid_size", 8),
            n_roombas=env_config.get("n_roombas", 2),
            render_mode=None,
            max_steps=env_config.get("max_steps", 500),
            verbose=False,
            log_dir=None,
        )
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.train_freq = train_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env_config.get("grid_size", 8) ** 2 + 2 * n_agents
        act_dim = 5

        self.dqns = [DQNNetwork(obs_dim, act_dim).to(self.device) for _ in range(n_agents)]
        self.target_dqns = [copy.deepcopy(dqn) for dqn in self.dqns]
        self.optimizers = [torch.optim.Adam(dqn.parameters(), lr=lr) for dqn in self.dqns]
        self.buffers = [ReplayBuffer(buffer_size) for _ in range(n_agents)]

        self.seed = seed

    def get_epsilon(self, step):
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_actions(self, obs, epsilon):
        actions = []
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        for i in range(self.n_agents):
            if random.random() < epsilon:
                actions.append(random.randint(0, 4))
            else:
                with torch.no_grad():
                    q = self.dqns[i](obs_t)
                    actions.append(q.argmax(dim=1).item())
        return np.array(actions)

    def train_step(self):
        for i in range(self.n_agents):
            if len(self.buffers[i]) < self.batch_size:
                return

            obs, actions, rewards, next_obs, dones = self.buffers[i].sample(self.batch_size)
            obs_t = torch.tensor(obs, device=self.device)
            actions_t = torch.tensor(actions, device=self.device).long()
            rewards_t = torch.tensor(rewards, device=self.device)
            next_obs_t = torch.tensor(next_obs, device=self.device)
            dones_t = torch.tensor(dones, device=self.device)

            # Current Q-values for chosen actions
            q_values = self.dqns[i](obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                next_q = self.target_dqns[i](next_obs_t).max(dim=1)[0]
                target = rewards_t + self.gamma * (1 - dones_t) * next_q

            loss = F.mse_loss(q_values, target)
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dqns[i].parameters(), 1.0)
            self.optimizers[i].step()

    def soft_update(self):
        for i in range(self.n_agents):
            for tp, sp in zip(self.target_dqns[i].parameters(), self.dqns[i].parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def train(self, total_timesteps, log_dir):
        logger = EpisodeLogger(log_dir)
        set_all_seeds(self.seed)
        obs, _ = self.env.reset(seed=self.seed)

        ep_reward = 0.0
        ep_count = 0
        step = 0

        while step < total_timesteps:
            epsilon = self.get_epsilon(step)
            actions = self.select_actions(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(actions)
            done = terminated or truncated

            # Store in each agent's buffer (same obs/reward, agent's own action)
            for i in range(self.n_agents):
                self.buffers[i].push(obs, actions[i], reward, next_obs, float(done))

            ep_reward += reward
            obs = next_obs
            step += 1

            if step >= self.warmup_steps and step % self.train_freq == 0:
                self.train_step()
                self.soft_update()

            if done:
                coverage_pct = self.env.coverage.sum() / (self.env.grid_size ** 2)
                ep_count += 1
                logger.log(ep_count, ep_reward, coverage_pct, self.env.steps,
                           self.env.total_collisions, terminated)

                if ep_count % 500 == 0:
                    print(f"[IDQN] Episode {ep_count} | Reward: {ep_reward:.1f} | "
                          f"Coverage: {coverage_pct*100:.1f}% | Eps: {epsilon:.3f} | Step: {step}")

                obs, _ = self.env.reset()
                ep_reward = 0.0

        return ep_count

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i, dqn in enumerate(self.dqns):
            torch.save(dqn.state_dict(), os.path.join(path, f"idqn_agent_{i}.pt"))

    def load(self, path):
        for i, dqn in enumerate(self.dqns):
            dqn.load_state_dict(torch.load(os.path.join(path, f"idqn_agent_{i}.pt"),
                                           weights_only=True))


def train(env_config, seed, total_timesteps, log_dir):
    trainer = IDQNTrainer(env_config, seed, n_agents=env_config.get("n_roombas", 2))
    trainer.train(total_timesteps, log_dir)
    model_path = os.path.join(log_dir, "models")
    trainer.save(model_path)
    trainer.env.close()
    return model_path


def evaluate(model_path, env_config, seed, n_episodes, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)

    n_agents = env_config.get("n_roombas", 2)
    obs_dim = env_config.get("grid_size", 8) ** 2 + 2 * n_agents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dqns = [DQNNetwork(obs_dim, 5).to(device) for _ in range(n_agents)]
    for i, dqn in enumerate(dqns):
        dqn.load_state_dict(torch.load(os.path.join(model_path, f"idqn_agent_{i}.pt"),
                                       weights_only=True, map_location=device))
        dqn.eval()

    env = MultiRoombaGrid(
        grid_size=env_config.get("grid_size", 8),
        n_roombas=n_agents,
        render_mode=None,
        max_steps=env_config.get("max_steps", 500),
        verbose=False,
        log_dir=None,
    )
    env.reset(seed=seed)

    def policy_fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        actions = []
        for dqn in dqns:
            with torch.no_grad():
                q = dqn(obs_t)
                actions.append(q.argmax(dim=1).item())
        return np.array(actions)

    metrics = evaluate_policy(policy_fn, env, n_episodes)
    save_metrics(metrics, os.path.join(log_dir, "eval_metrics.json"))
    env.close()
    return metrics
