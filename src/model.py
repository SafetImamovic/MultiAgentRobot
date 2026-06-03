import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import config


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, n_agents, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + n_agents * n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Q-value output

    def forward(self, state, actions_onehot):
        # state: global_state batch, actions_onehot: batch x (n_agents * n_actions)
        x = torch.cat([state, actions_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def sample_batch(replay_buffer, batch_size):
    batch = random.sample(replay_buffer, batch_size)

    obs = [
        torch.stack([b["obs"][i] for b in batch], dim=0)
        for i in range(config.NUM_AGENTS)
    ]
    actions = [
        torch.tensor([b["actions"][i] for b in batch], dtype=torch.long)
        for i in range(config.NUM_AGENTS)
    ]
    rewards = [
        torch.tensor([b["rewards"][i] for b in batch], dtype=torch.float32)
        for i in range(config.NUM_AGENTS)
    ]
    next_obs = [
        torch.stack([b["next_obs"][i] for b in batch], dim=0)
        for i in range(config.NUM_AGENTS)
    ]
    global_state = torch.tensor([b["global_state"] for b in batch], dtype=torch.float32)
    next_global_state = torch.tensor(
        [b["next_global_state"] for b in batch], dtype=torch.float32
    )
    done = torch.tensor([b["done"] for b in batch], dtype=torch.float32)

    return obs, actions, rewards, next_obs, global_state, next_global_state, done


def actions_to_onehot(actions, n_actions):
    # actions: list of tensors for each agent, shape = (batch,)
    batch_size = actions[0].shape[0]
    n_agents = len(actions)
    onehot = torch.zeros(batch_size, n_agents * n_actions)
    for i, a in enumerate(actions):
        onehot[torch.arange(batch_size), i * n_actions + a] = 1
    return onehot




def train_step(
    dqns,
    critic,
    target_dqns,
    target_critic,
    replay_buffer,
    optimizer_dqns,
    optimizer_critic,
    gamma=0.99,
    batch_size=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Sample batch -----
    obs, actions, rewards, next_obs, global_state, next_global_state, done = (
        sample_batch(replay_buffer, batch_size)
    )

    # Move everything to device
    obs = [o.to(device) for o in obs]
    next_obs = [no.to(device) for no in next_obs]
    actions = [a.to(device) for a in actions]
    rewards = [r.to(device) for r in rewards]
    global_state = global_state.to(device)
    next_global_state = next_global_state.to(device)
    done = done.to(device)

    # Move models to device
    for i in range(len(dqns)):
        dqns[i].to(device)
        target_dqns[i].to(device)
    critic.to(device)
    target_critic.to(device)

    # ----- Update Independent DQNs -----
    for i, dqn in enumerate(dqns):
        q_values = dqn(obs[i])
        q_values = q_values.gather(1, actions[i].unsqueeze(1)).squeeze(1)  # Q(s,a)

        with torch.no_grad():
            next_q = target_dqns[i](next_obs[i])
            max_next_q = next_q.max(dim=1)[0]
            target = rewards[i] + gamma * (1 - done) * max_next_q

        loss = F.mse_loss(q_values, target)

        optimizer_dqns[i].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)  # Clip gradients
        optimizer_dqns[i].step()

    # ----- Update Centralized Critic -----
    actions_onehot = actions_to_onehot(actions, n_actions=config.N_ACTIONS).to(device)
    next_actions_onehot = actions_to_onehot(
        [dqn(no).argmax(dim=1) for dqn, no in zip(target_dqns, next_obs)],
        n_actions=config.N_ACTIONS,
    ).to(device)

    critic_values = critic(global_state, actions_onehot)

    total_reward = (
        torch.stack(rewards, dim=1).sum(dim=1).unsqueeze(1)
    )  # Sum of all agent rewards

    with torch.no_grad():
        next_actions = [dqn(no).argmax(dim=1) for dqn, no in zip(target_dqns, next_obs)]
        next_actions_onehot = actions_to_onehot(
            next_actions, n_actions=config.N_ACTIONS
        ).to(device)
        target_values = total_reward + gamma * (1 - done.unsqueeze(1)) * target_critic(
            next_global_state, next_actions_onehot
        )

    loss_critic = F.mse_loss(critic_values, target_values)

    optimizer_critic.zero_grad()
    loss_critic.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)  # Clip gradients
    optimizer_critic.step()
