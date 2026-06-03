import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from model import DQN, CentralizedCritic, train_step
from env import SobaGridEnv
from utils import PygameRenderer
import random
import numpy as np
import time
import pygame
from collections import deque
import config
import argparse

# ---- parse CLI args ----
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without rendering")
args = parser.parse_args()
HEADLESS = args.headless
if HEADLESS:
    print("Running in headless mode (no rendering)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- replay buffer ----
BUFFER_SIZE = 50000
replay_buffer = deque(maxlen=BUFFER_SIZE)

# ---- networks ----
dqns = [
    DQN(input_dim=config.OBS_DIM, output_dim=config.N_ACTIONS).to(device)
    for _ in range(config.NUM_AGENTS)
]
target_dqns = [
    DQN(input_dim=config.OBS_DIM, output_dim=config.N_ACTIONS).to(device)
    for _ in range(config.NUM_AGENTS)
]
for target, net in zip(target_dqns, dqns):
    target.load_state_dict(net.state_dict())
    target.to(device)  # ensure target is also on GPU

critic = CentralizedCritic(
    state_dim=config.GLOBAL_STATE_DIM,
    n_agents=config.NUM_AGENTS,
    n_actions=config.N_ACTIONS,
).to(device)
target_critic = CentralizedCritic(
    state_dim=config.GLOBAL_STATE_DIM,
    n_agents=config.NUM_AGENTS,
    n_actions=config.N_ACTIONS,
).to(device)
target_critic.load_state_dict(critic.state_dict())

# ---- optimizers ----
optimizer_dqns = [optim.Adam(dqn.parameters(), lr=3e-4) for dqn in dqns]
optimizer_critic = optim.Adam(critic.parameters(), lr=3e-4)

# ---- hyperparams ----
BATCH_SIZE = 256
GAMMA = 0.99  # Increased from 0.97 for better long-term planning
TAU = 0.005  # Increased from 0.001 for faster target network updates
WARMUP_STEPS = 1000  # Don't train until we have diverse experiences


# ---- helper functions ----


def korak(prvi_korak: bool, device=None, env=None, epsilon=0.05):
    """Epsilon-greedy actions from DQN."""

    if prvi_korak:
        return [
            random.randint(0, config.N_ACTIONS - 1)
            for _ in range(config.NUM_AGENTS)
        ]

    # Not first step - use DQN for all agents
    combined_coverage = get_combined_coverage(env)
    obs_list = get_obs_list(env, combined_coverage)

    actions = []
    for i, dqn in enumerate(dqns):
        obs = obs_list[i].unsqueeze(0)

        if device is not None:
            obs = obs.to(device)

        # Epsilon-greedy: small exploration
        if random.random() < epsilon:
            actions.append(random.randint(0, config.N_ACTIONS - 1))
        else:
            with torch.no_grad():
                q_values = dqn(obs)
                actions.append(q_values.argmax().item())

    return actions


def get_obs_list(env, combined_coverage):
    obs_list = []
    H, W = env.state.grid.shape
    for agent in env.state.agents:
        latest_step = agent.koraci[-1]
        x, y = latest_step["pozicija_x"], latest_step["pozicija_y"]
        coverage_channel = (1.0 - combined_coverage).astype(np.float32)  # Invert: 1.0 = uncleaned
        # coverage_channel = (combined_coverage).astype(np.float32)  # Invert: 1.0 = uncleaned
        walls_channel = (env.state.init_grid == 1).astype(np.float32)  # 1.0 = wall only
        agent_channel = np.zeros((H, W), dtype=np.float32)
        agent_channel[x, y] = 1.0
        obs = np.stack([coverage_channel, walls_channel, agent_channel], axis=0)
        obs_list.append(torch.tensor(obs.flatten(), dtype=torch.float32))
    return obs_list


def get_combined_coverage(env):
    coverage_maps = [
        np.array(agent.koraci[-1]["coverage_map"]) for agent in env.state.agents
    ]
    return np.maximum.reduce(coverage_maps)


def get_global_state(env, combined_coverage):
    positions = np.concatenate(
        [
            np.array(
                [a.koraci[-1]["pozicija_x"], a.koraci[-1]["pozicija_y"]],
                dtype=np.float32,
            )
            for a in env.state.agents
        ]
    )
    global_state = np.concatenate(
        [
            combined_coverage.flatten(),
            np.array(env.state.init_grid, dtype=np.float32).flatten(),
            positions,
        ]
    )
    return global_state


def soft_update(target, source, tau=TAU):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)


# ---- main loop ----
if __name__ == "__main__":
    all_rewards = []
    all_coverage = []

    os.makedirs("epizode", exist_ok=True)

    print(torch.version.cuda)

    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print(torch.cuda.current_device())

        print(torch.cuda.get_device_name(0))

    for episode in range(1, config.MAX_EPISODES + 1):
        episode_reward = np.zeros(config.NUM_AGENTS)
        env = SobaGridEnv(episode)
        state = env.reset()

        renderer = None
        if not HEADLESS:
            renderer = PygameRenderer(env)
            clock = pygame.time.Clock()

        done = False
        last_step_time = time.time()

        while True:
            now = time.time()

            # --- handle pygame events ---
            if not HEADLESS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

            if HEADLESS:
                config.STEP_INTERVAL = 0.0

            # --- simulation step ---
            if not done and (now - last_step_time) >= config.STEP_INTERVAL:
                # Epsilon decay should ideally be per episode or per step. 
                # Here it is per step, which is fine if config.EPSILON_DECAY is tuned for it.

                # Get current observation before taking action
                combined_coverage = get_combined_coverage(env)
                obs_list = get_obs_list(env, combined_coverage)
                global_state = get_global_state(env, combined_coverage)

                actions = korak(env.prvi_korak, device=device, env=env)

                if env.prvi_korak:
                    env.prvi_korak = False

                _, rewards, done = env.step(actions)

                episode_reward += np.array(rewards)

                # Get observation after taking action
                next_combined_coverage = get_combined_coverage(env)
                next_obs_list = get_obs_list(env, next_combined_coverage)
                next_global_state = get_global_state(env, next_combined_coverage)

                rewards_list = [
                    agent.koraci[-1]["nagrada"] for agent in env.state.agents
                ]
                done_flag = done

                replay_buffer.append(
                    {
                        "obs": [o.clone() for o in obs_list],
                        "actions": actions,
                        "rewards": rewards_list,
                        "next_obs": [o.clone() for o in next_obs_list],
                        "global_state": global_state,
                        "next_global_state": next_global_state,
                        "done": done_flag,
                    }
                )

                # --- train if enough samples and after warmup ---
                if len(replay_buffer) >= max(BATCH_SIZE, WARMUP_STEPS):
                    train_step(
                        dqns,
                        critic,
                        target_dqns,
                        target_critic,
                        replay_buffer,
                        optimizer_dqns,
                        optimizer_critic,
                        gamma=GAMMA,
                        batch_size=BATCH_SIZE,
                    )
                    for dqn, target in zip(dqns, target_dqns):
                        soft_update(target, dqn)
                    soft_update(target_critic, critic)

                # --- rendering ---
                if not HEADLESS:
                    renderer.render(agent_rewards=rewards_list)
                    renderer.update_only()
                    pygame.display.flip()
                    clock.tick(config.FPS)

                last_step_time = now

            # --- throttle for headless ---
            if HEADLESS:
                time.sleep(config.STEP_INTERVAL)

            if done:
                all_rewards.append(episode_reward)
                coverage_pct = (get_combined_coverage(env).sum() / (
                            env.state.grid.shape[0] * env.state.grid.shape[1])) * 100
                all_coverage.append(coverage_pct)

                # Calculate rolling average over last 10 episodes
                window = 10
                if len(all_coverage) >= window:
                    avg_coverage = np.mean(all_coverage[-window:])
                    print(f"Episode {episode} | Coverage: {coverage_pct:.1f}% | Avg(10): {avg_coverage:.1f}% | Steps: {env.state.step_count} | Rewards: {all_rewards[-1]}")
                else:
                    print(f"Episode {episode} | Coverage: {coverage_pct:.1f}% | Steps: {env.state.step_count} | Rewards: {all_rewards[-1]}")
                break

    # shape: (num_episodes, num_agents)

    # Save DQNs
    for i, dqn in enumerate(dqns):
        torch.save(dqn.state_dict(), f"dqn_agent_{i}.pt")

    # Save centralized critic
    torch.save(critic.state_dict(), "central_critic.pt")

    all_rewards_array = np.array(all_rewards)

    num_agents = all_rewards_array.shape[1]
    episodes = np.arange(1, len(all_rewards_array) + 1)

    plt.figure(figsize=(10, 6))

    for i in range(num_agents):
        plt.plot(episodes, all_rewards_array[:, i], label=f"Agent {i + 1}")

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Agent Rewards Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()
