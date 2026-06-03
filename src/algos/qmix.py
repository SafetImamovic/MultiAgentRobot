import os
import sys
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roomba_env import MultiRoombaGrid
from algos.common import (
    set_all_seeds, DQNNetwork, ReplayBuffer,
    evaluate_policy, EpisodeLogger, save_metrics,
)


# ---------------------
# QMIX Mixing Network
# ---------------------

class QMIXMixer(nn.Module):
    """
    Monotonic mixing network with hypernetwork-generated weights.
    Combines per-agent Q-values into Q_tot while enforcing
    dQ_tot/dQ_i >= 0 via abs() on weights.
    """

    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Hypernetwork: state -> first-layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork: state -> second-layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state):
        """
        Args:
            agent_qs: [batch, n_agents] — per-agent Q-values
            state: [batch, state_dim] — global state
        Returns:
            q_tot: [batch] — total Q-value
        """
        batch_size = agent_qs.size(0)

        # First layer: monotonic via abs()
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1).squeeze(-1)


# ---------------------
# QMIX Trainer
# ---------------------

class QMIXTrainer:

    def __init__(self, env_config, seed, n_agents=2, lr=5e-4, buffer_size=100_000,
                 batch_size=32, gamma=0.99, tau=0.005, embed_dim=32,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=500_000,
                 warmup_steps=1000, train_freq=4):
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
        state_dim = obs_dim  # Full observability: global obs == state
        act_dim = 5

        # Per-agent Q-networks
        self.agent_nets = [DQNNetwork(obs_dim, act_dim).to(self.device) for _ in range(n_agents)]
        self.target_agent_nets = [copy.deepcopy(net) for net in self.agent_nets]

        # Mixing network
        self.mixer = QMIXMixer(n_agents, state_dim, embed_dim).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)

        # Single optimizer for all agent nets + mixer
        all_params = list(self.mixer.parameters())
        for net in self.agent_nets:
            all_params += list(net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

        # Joint replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.seed = seed

    def get_epsilon(self, step):
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_actions(self, obs, epsilon):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        for i in range(self.n_agents):
            if random.random() < epsilon:
                actions.append(random.randint(0, 4))
            else:
                with torch.no_grad():
                    q = self.agent_nets[i](obs_t)
                    actions.append(q.argmax(dim=1).item())
        return np.array(actions)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.tensor(obs, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).long()
        rewards_t = torch.tensor(rewards, device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Per-agent Q-values for chosen actions
        agent_qs = []
        for i in range(self.n_agents):
            q = self.agent_nets[i](obs_t)
            agent_q = q.gather(1, actions_t[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(agent_q)
        agent_qs = torch.stack(agent_qs, dim=1)  # [batch, n_agents]

        # Q_tot via mixing network
        q_tot = self.mixer(agent_qs, obs_t)

        # Target Q_tot
        with torch.no_grad():
            target_agent_qs = []
            for i in range(self.n_agents):
                target_q = self.target_agent_nets[i](next_obs_t).max(dim=1)[0]
                target_agent_qs.append(target_q)
            target_agent_qs = torch.stack(target_agent_qs, dim=1)

            target_q_tot = self.target_mixer(target_agent_qs, next_obs_t)
            targets = rewards_t + self.gamma * (1 - dones_t) * target_q_tot

        loss = F.mse_loss(q_tot, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.mixer.parameters()) + sum([list(n.parameters()) for n in self.agent_nets], []),
            10.0,
        )
        self.optimizer.step()

    def soft_update(self):
        for i in range(self.n_agents):
            for tp, sp in zip(self.target_agent_nets[i].parameters(), self.agent_nets[i].parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.target_mixer.parameters(), self.mixer.parameters()):
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

            self.buffer.push(obs, actions, reward, next_obs, float(done))
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
                    print(f"[QMIX] Episode {ep_count} | Reward: {ep_reward:.1f} | "
                          f"Coverage: {coverage_pct*100:.1f}% | Eps: {epsilon:.3f} | Step: {step}")

                obs, _ = self.env.reset()
                ep_reward = 0.0

        return ep_count

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i, net in enumerate(self.agent_nets):
            torch.save(net.state_dict(), os.path.join(path, f"qmix_agent_{i}.pt"))
        torch.save(self.mixer.state_dict(), os.path.join(path, "qmix_mixer.pt"))

    def load(self, path):
        for i, net in enumerate(self.agent_nets):
            net.load_state_dict(torch.load(os.path.join(path, f"qmix_agent_{i}.pt"),
                                           weights_only=True))
        self.mixer.load_state_dict(torch.load(os.path.join(path, "qmix_mixer.pt"),
                                              weights_only=True))


# ---------------------
# Module interface
# ---------------------

def train(env_config, seed, total_timesteps, log_dir):
    trainer = QMIXTrainer(env_config, seed, n_agents=env_config.get("n_roombas", 2))
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

    agent_nets = [DQNNetwork(obs_dim, 5).to(device) for _ in range(n_agents)]
    for i, net in enumerate(agent_nets):
        net.load_state_dict(torch.load(os.path.join(model_path, f"qmix_agent_{i}.pt"),
                                       weights_only=True, map_location=device))
        net.eval()

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
        for net in agent_nets:
            with torch.no_grad():
                q = net(obs_t)
                actions.append(q.argmax(dim=1).item())
        return np.array(actions)

    metrics = evaluate_policy(policy_fn, env, n_episodes)
    save_metrics(metrics, os.path.join(log_dir, "eval_metrics.json"))
    env.close()
    return metrics
