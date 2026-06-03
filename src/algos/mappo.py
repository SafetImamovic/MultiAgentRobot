import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roomba_env import MultiRoombaGrid
from algos.common import set_all_seeds, evaluate_policy, EpisodeLogger, save_metrics


# ---------------------
# Networks
# ---------------------

class MAPPOActor(nn.Module):
    """Per-agent policy with parameter sharing. Agent ID appended to obs."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return Categorical(logits=self.net(obs))

    def get_action_and_logprob(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, obs, actions):
        dist = self.forward(obs)
        return dist.log_prob(actions), dist.entropy()


class MAPPOCritic(nn.Module):
    """Centralized value function."""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ---------------------
# Rollout Buffer
# ---------------------

class RolloutBuffer:
    """Stores rollout data for all agents."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self, device):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
            torch.tensor(np.array(self.logprobs), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.dones), dtype=torch.float32, device=device),
            torch.tensor(np.array(self.values), dtype=torch.float32, device=device),
        )

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.obs)


# ---------------------
# MAPPO Trainer
# ---------------------

class MAPPOTrainer:

    def __init__(self, env_config, seed, n_agents=2, actor_lr=5e-4, critic_lr=5e-4,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                 n_epochs=10, n_steps=2048, batch_size=256):
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
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        grid_size = env_config.get("grid_size", 8)
        base_obs_dim = grid_size ** 2 + 2 * n_agents
        # Append one-hot agent ID to differentiate agents with shared params
        actor_obs_dim = base_obs_dim + n_agents
        state_dim = base_obs_dim  # Critic sees global state

        self.base_obs_dim = base_obs_dim
        self.actor_obs_dim = actor_obs_dim

        self.actor = MAPPOActor(actor_obs_dim, 5).to(self.device)
        self.critic = MAPPOCritic(state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffers = [RolloutBuffer() for _ in range(n_agents)]

    def _agent_obs(self, obs, agent_id):
        """Append one-hot agent ID to observation."""
        agent_onehot = np.zeros(self.n_agents, dtype=np.float32)
        agent_onehot[agent_id] = 1.0
        return np.concatenate([obs, agent_onehot])

    def collect_rollout(self, logger=None):
        """Collect n_steps of experience for all agents, logging completed episodes."""
        if not hasattr(self, '_rollout_obs'):
            self._rollout_obs, _ = self.env.reset()
            self._ep_reward = 0.0
        obs = self._rollout_obs
        steps_collected = 0

        while steps_collected < self.n_steps:
            # Get value estimate from critic (same global state for all agents)
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                value = self.critic(state_t).item()

            # Get actions for all agents
            actions = []
            logprobs = []
            for i in range(self.n_agents):
                agent_obs = self._agent_obs(obs, i)
                obs_t = torch.tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, logprob = self.actor.get_action_and_logprob(obs_t)
                actions.append(action.item())
                logprobs.append(logprob.item())

            joint_actions = np.array(actions)
            next_obs, reward, terminated, truncated, _ = self.env.step(joint_actions)
            done = terminated or truncated
            self._ep_reward += reward

            # Store transition for each agent
            per_agent_reward = reward / self.n_agents
            for i in range(self.n_agents):
                agent_obs = self._agent_obs(obs, i)
                self.buffers[i].add(agent_obs, actions[i], logprobs[i],
                                    per_agent_reward, float(done), value)

            obs = next_obs
            steps_collected += 1

            if done:
                # Log completed episode
                if logger is not None:
                    if not hasattr(self, '_ep_count'):
                        self._ep_count = 0
                    self._ep_count += 1
                    coverage_pct = self.env.coverage.sum() / (self.env.grid_size ** 2)
                    logger.log(self._ep_count, self._ep_reward, coverage_pct,
                               self.env.steps, self.env.total_collisions, terminated)
                obs, _ = self.env.reset()
                self._ep_reward = 0.0

        self._rollout_obs = obs

        # Bootstrap value for last state
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(state_t).item()

        return last_value

    def compute_gae(self, rewards, values, dones, last_value):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values)
        return advantages, returns

    def update(self, last_value):
        """PPO update using collected rollout data from all agents."""
        # Merge all agent buffers
        all_obs, all_actions, all_logprobs = [], [], []
        all_advantages, all_returns = [], []

        for i in range(self.n_agents):
            obs_np = np.array(self.buffers[i].obs)
            actions_np = np.array(self.buffers[i].actions)
            logprobs_np = np.array(self.buffers[i].logprobs)
            rewards_np = np.array(self.buffers[i].rewards)
            dones_np = np.array(self.buffers[i].dones)
            values_np = np.array(self.buffers[i].values)

            advantages, returns = self.compute_gae(rewards_np, values_np, dones_np, last_value)

            all_obs.append(obs_np)
            all_actions.append(actions_np)
            all_logprobs.append(logprobs_np)
            all_advantages.append(advantages)
            all_returns.append(returns)

        # Concatenate across agents
        all_obs = torch.tensor(np.concatenate(all_obs), dtype=torch.float32, device=self.device)
        all_actions = torch.tensor(np.concatenate(all_actions), dtype=torch.long, device=self.device)
        all_logprobs = torch.tensor(np.concatenate(all_logprobs), dtype=torch.float32, device=self.device)
        all_advantages = torch.tensor(np.concatenate(all_advantages), dtype=torch.float32, device=self.device)
        all_returns = torch.tensor(np.concatenate(all_returns), dtype=torch.float32, device=self.device)

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        n_samples = len(all_obs)

        for _ in range(self.n_epochs):
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                b_obs = all_obs[batch_idx]
                b_actions = all_actions[batch_idx]
                b_old_logprobs = all_logprobs[batch_idx]
                b_advantages = all_advantages[batch_idx]
                b_returns = all_returns[batch_idx]

                # Actor loss (PPO clipped)
                new_logprobs, entropy = self.actor.evaluate_actions(b_obs, b_actions)
                ratio = torch.exp(new_logprobs - b_old_logprobs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Critic loss (global state = base obs without agent ID)
                b_state = b_obs[:, :self.base_obs_dim]
                values = self.critic(b_state)
                critic_loss = F.mse_loss(values, b_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        # Clear buffers
        for buf in self.buffers:
            buf.clear()

    def train(self, total_timesteps, log_dir):
        logger = EpisodeLogger(log_dir)
        set_all_seeds(self.seed)
        self.env.reset(seed=self.seed)

        total_steps = 0
        update_count = 0

        while total_steps < total_timesteps:
            last_value = self.collect_rollout(logger=logger)
            self.update(last_value)

            total_steps += self.n_steps
            update_count += 1

            if update_count % 10 == 0:
                ep_count = getattr(self, '_ep_count', 0)
                metrics = self._quick_eval(5)
                print(f"[MAPPO] Steps: {total_steps} | Episodes: {ep_count} | "
                      f"Success: {metrics['success_rate']:.1f}% | "
                      f"Reward: {metrics['avg_reward']:.1f} | "
                      f"Coverage: {metrics['avg_coverage']:.1f}%")

        return update_count

    def _greedy_actions(self, obs):
        actions = []
        for i in range(self.n_agents):
            agent_obs = self._agent_obs(obs, i)
            obs_t = torch.tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                dist = self.actor(obs_t)
                actions.append(dist.probs.argmax(dim=1).item())
        return np.array(actions)

    def _quick_eval(self, n_episodes):
        results = {"success_rate": 0, "avg_reward": 0, "avg_coverage": 0}
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                actions = self._greedy_actions(obs)
                obs, reward, terminated, truncated, _ = self.env.step(actions)
                ep_reward += reward
                done = terminated or truncated
            coverage = self.env.coverage.sum() / (self.env.grid_size ** 2)
            results["success_rate"] += float(terminated)
            results["avg_reward"] += ep_reward
            results["avg_coverage"] += coverage * 100
        for k in results:
            results[k] /= n_episodes
        results["success_rate"] *= 100
        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "mappo_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "mappo_critic.pt"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "mappo_actor.pt"),
                                              weights_only=True))
        self.critic.load_state_dict(torch.load(os.path.join(path, "mappo_critic.pt"),
                                               weights_only=True))


# ---------------------
# Module interface
# ---------------------

def train(env_config, seed, total_timesteps, log_dir):
    trainer = MAPPOTrainer(env_config, seed, n_agents=env_config.get("n_roombas", 2))
    trainer.train(total_timesteps, log_dir)
    model_path = os.path.join(log_dir, "models")
    trainer.save(model_path)
    trainer.env.close()
    return model_path


def evaluate(model_path, env_config, seed, n_episodes, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    set_all_seeds(seed)

    n_agents = env_config.get("n_roombas", 2)
    grid_size = env_config.get("grid_size", 8)
    base_obs_dim = grid_size ** 2 + 2 * n_agents
    actor_obs_dim = base_obs_dim + n_agents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = MAPPOActor(actor_obs_dim, 5).to(device)
    actor.load_state_dict(torch.load(os.path.join(model_path, "mappo_actor.pt"),
                                     weights_only=True, map_location=device))
    actor.eval()

    env = MultiRoombaGrid(
        grid_size=grid_size,
        n_roombas=n_agents,
        render_mode=None,
        max_steps=env_config.get("max_steps", 500),
        verbose=False,
        log_dir=None,
    )
    env.reset(seed=seed)

    def policy_fn(obs):
        actions = []
        for i in range(n_agents):
            agent_onehot = np.zeros(n_agents, dtype=np.float32)
            agent_onehot[i] = 1.0
            agent_obs = np.concatenate([obs, agent_onehot])
            obs_t = torch.tensor(agent_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = actor(obs_t)
                actions.append(dist.probs.argmax(dim=1).item())
        return np.array(actions)

    metrics = evaluate_policy(policy_fn, env, n_episodes)
    save_metrics(metrics, os.path.join(log_dir, "eval_metrics.json"))
    env.close()
    return metrics
