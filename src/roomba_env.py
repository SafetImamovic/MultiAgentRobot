import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import pygame
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

# =======================
# Global configuration
# =======================
GRID_SIZE = 8
N_ROOMBAS = 2
CELL_SIZE = 80
COLLISION_PENALTY = -0.5  # Penalty for roomba-roomba collisions
WALL_COLLISION_PENALTY = -0.3  # Penalty for hitting walls/borders
REWARD_PER_TILE = +1.0
COMPLETION_BONUS = +200.0
REVISIT_PENALTY = -0.05  # Penalty for visiting already cleaned tiles
WAIT_PENALTY = -0.1  # Penalty for staying in place
LOG_DIR = "logs_v2"
PLOT_DIR = "plots_v2"
SAVE_PATH = "models_v2/"
TENSORBOARD_LOG = "./ppo_roomba_tb_v2"
RENDER_MODE = "human"
TOTAL_TIMESTEPS = 10_000_000
# MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DEVICE = "cpu"
EPISODES_TO_RENDER = 5
CHECKPOINT_FREQ = 5_000


# =======================
# Environment
# =======================
class MultiRoombaGrid(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=GRID_SIZE, n_roombas=N_ROOMBAS, render_mode=RENDER_MODE,
                 log_dir=LOG_DIR, max_steps=500, verbose=False):
        super().__init__()

        self.grid_size = grid_size
        self.n = n_roombas
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.verbose = verbose

        # Action: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_space = spaces.MultiDiscrete([5] * self.n)

        # Observation: flattened coverage map + normalized positions
        obs_dim = grid_size * grid_size + 2 * self.n
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)

        # Rendering
        self.cell_size = CELL_SIZE
        self.screen = None
        self.clock = None

        # Logging
        self.log_dir = log_dir
        self.episode_count = 0
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.csv_file = os.path.join(log_dir, "episode_data.csv") if log_dir else None
        if self.csv_file and not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward", "coverage_pct", "steps",
                                 "total_collisions", "terminated"])

        # Episode stats
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.coverage = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        self.positions = []

        # Random initial positions
        for _ in range(self.n):
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            self.positions.append([x, y])
            self.coverage[y, x] = 1.0

        self.steps = 0
        self.episode_count += 1
        self.total_reward = 0.0
        self.total_collisions = 0

        return self._get_obs(), {}

    def step(self, actions):
        step_reward = 0.0
        new_positions = []

        # Execute moves and detect wall collisions
        for i, a in enumerate(actions):
            x, y = self.positions[i]

            if a == 0:  # up
                if y > 0:
                    y -= 1
                else:
                    step_reward += WALL_COLLISION_PENALTY  # Hit top wall
            elif a == 1:  # down
                if y < self.grid_size - 1:
                    y += 1
                else:
                    step_reward += WALL_COLLISION_PENALTY  # Hit bottom wall
            elif a == 2:  # left
                if x > 0:
                    x -= 1
                else:
                    step_reward += WALL_COLLISION_PENALTY  # Hit left wall
            elif a == 3:  # right
                if x < self.grid_size - 1:
                    x += 1
                else:
                    step_reward += WALL_COLLISION_PENALTY  # Hit right wall
            elif a == 4:  # stay
                step_reward += WAIT_PENALTY

            new_positions.append([x, y])

        # Collision detection (roomba-roomba)
        collisions = 0

        position_counts = {}

        for pos in new_positions:
            key = tuple(pos)
            position_counts[key] = position_counts.get(key, 0) + 1

        # Count unique collision events (cells with 2+ roombas)
        for count in position_counts.values():
            if count > 1:
                collisions += 1

        self.positions = new_positions
        self.total_collisions += collisions

        # Reward for cleaning new tiles, penalize revisiting cleaned tiles
        new_tiles = 0
        revisits = 0
        for x, y in self.positions:
            if self.coverage[y, x] == 0:
                new_tiles += 1
                self.coverage[y, x] = 1.0
            else:
                revisits += 1

        step_reward += REWARD_PER_TILE * new_tiles
        step_reward += REVISIT_PENALTY * revisits
        step_reward += COLLISION_PENALTY * collisions

        self.total_reward += step_reward
        self.steps += 1

        # Termination conditions
        coverage_pct = self.coverage.sum() / (self.grid_size ** 2)

        terminated = coverage_pct == 1.0

        if terminated:
            step_reward += COMPLETION_BONUS
            self.total_reward += COMPLETION_BONUS

        truncated = self.steps >= self.max_steps

        # Logging (only at episode end)
        if terminated or truncated:
            self._log_episode(coverage_pct, terminated)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), step_reward, terminated, truncated, {}

    def _get_obs(self):
        flat_map = self.coverage.flatten()
        flat_pos = np.array(self.positions, dtype=np.float32).flatten() / self.grid_size
        return np.concatenate([flat_map, flat_pos])

    def render(self):
        if self.screen is None:
            pygame.init()

            width = self.grid_size * self.cell_size

            self.screen = pygame.display.set_mode((width, width))

            pygame.display.set_caption(f"Multi-Roomba Grid - {self.n} Roombas")

            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((40, 40, 40))

        # Draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = (220, 240, 220) if self.coverage[y, x] else (25, 25, 25)
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

        # Draw roombas with collision indication
        position_counts = {}
        for pos in self.positions:
            key = tuple(pos)
            position_counts[key] = position_counts.get(key, 0) + 1

        for x, y in self.positions:
            color = (255, 50, 50) if position_counts[(x, y)] > 1 else (50, 120, 255)
            center = (x * self.cell_size + self.cell_size // 2,
                      y * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.screen, color, center, self.cell_size // 3)
            pygame.draw.circle(self.screen, (255, 255, 255), center, self.cell_size // 3, 2)

        # Display stats
        if self.verbose:
            coverage_pct = self.coverage.sum() / (self.grid_size ** 2) * 100
            font = pygame.font.Font(None, 24)
            stats_text = f"Episode: {self.episode_count} | Step: {self.steps} | Coverage: {coverage_pct:.1f}%"
            text_surf = font.render(stats_text, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, 10))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(30)

    def _log_episode(self, coverage_pct, terminated):
        if self.verbose:
            status = "COMPLETE" if terminated else "TRUNCATED"
            print(f"[Ep {self.episode_count}] {status} | Reward: {self.total_reward:.1f} | "
                  f"Coverage: {coverage_pct * 100:.1f}% | Steps: {self.steps} | "
                  f"Collisions: {self.total_collisions}")

        if self.csv_file:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_count,
                    self.total_reward,
                    coverage_pct,
                    self.steps,
                    self.total_collisions,
                    terminated
                ])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


# =======================
# Plotting functions
# =======================
def plot_training_metrics(csv_path, save_dir):
    """Generate and save training metric plots from CSV data."""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Total Reward
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['episode'], df['total_reward'], linewidth=1, alpha=0.6, label='Episode Reward')
    window = min(50, len(df) // 10)
    if window > 1:
        rolling_avg = df['total_reward'].rolling(window=window).mean()
        ax.plot(df['episode'], rolling_avg, linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Training Progress: Total Reward per Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reward_progress_{timestamp}.png'), dpi=150)
    plt.close()

    # 2. Coverage Percentage
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['episode'], df['coverage_pct'] * 100, linewidth=1, alpha=0.6, label='Coverage %')
    if window > 1:
        rolling_avg = (df['coverage_pct'] * 100).rolling(window=window).mean()
        ax.plot(df['episode'], rolling_avg, linewidth=2, color='green', label=f'{window}-Episode Moving Avg')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Training Progress: Grid Coverage per Episode', fontsize=14, fontweight='bold')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'coverage_progress_{timestamp}.png'), dpi=150)
    plt.close()

    # 3. Steps to Completion
    fig, ax = plt.subplots(figsize=(12, 6))
    successful = df[df['terminated'] == True]
    if len(successful) > 0:
        ax.plot(successful['episode'], successful['steps'], linewidth=1, alpha=0.6,
                marker='o', markersize=3, label='Steps to Complete')
        if len(successful) > 10:
            rolling_avg = successful['steps'].rolling(window=10).mean()
            ax.plot(successful['episode'], rolling_avg, linewidth=2, color='orange',
                    label='10-Episode Moving Avg')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Steps', fontsize=12)
        ax.set_title('Efficiency: Steps to 100% Coverage (Successful Episodes Only)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No successful episodes yet', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Efficiency: Steps to 100% Coverage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'efficiency_progress_{timestamp}.png'), dpi=150)
    plt.close()

    # 4. Collision Statistics
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['episode'], df['total_collisions'], linewidth=1, alpha=0.6, label='Collisions')
    if window > 1:
        rolling_avg = df['total_collisions'].rolling(window=window).mean()
        ax.plot(df['episode'], rolling_avg, linewidth=2, color='red', label=f'{window}-Episode Moving Avg')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Collisions', fontsize=12)
    ax.set_title('Coordination: Total Collisions per Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'collisions_progress_{timestamp}.png'), dpi=150)
    plt.close()

    # 5. Success Rate
    fig, ax = plt.subplots(figsize=(12, 6))
    success_window = min(100, len(df) // 5)
    if success_window > 1:
        success_rate = df['terminated'].rolling(window=success_window).mean() * 100
        ax.plot(df['episode'], success_rate, linewidth=2, color='purple',
                label=f'Success Rate ({success_window}-Episode Window)')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Learning Progress: Success Rate (100% Coverage Achievement)',
                     fontsize=14, fontweight='bold')
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Success')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough episodes for success rate', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Learning Progress: Success Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'success_rate_{timestamp}.png'), dpi=150)
    plt.close()

    # 6. Combined Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Summary Dashboard', fontsize=16, fontweight='bold')

    # Reward
    axes[0, 0].plot(df['episode'], df['total_reward'], linewidth=1, alpha=0.6)
    if window > 1:
        axes[0, 0].plot(df['episode'], df['total_reward'].rolling(window=window).mean(),
                        linewidth=2, color='red')
    axes[0, 0].set_title('Total Reward', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Coverage
    axes[0, 1].plot(df['episode'], df['coverage_pct'] * 100, linewidth=1, alpha=0.6)
    if window > 1:
        axes[0, 1].plot(df['episode'], (df['coverage_pct'] * 100).rolling(window=window).mean(),
                        linewidth=2, color='green')
    axes[0, 1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Coverage %', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Coverage (%)')
    axes[0, 1].grid(True, alpha=0.3)

    # Collisions
    axes[1, 0].plot(df['episode'], df['total_collisions'], linewidth=1, alpha=0.6)
    if window > 1:
        axes[1, 0].plot(df['episode'], df['total_collisions'].rolling(window=window).mean(),
                        linewidth=2, color='red')
    axes[1, 0].set_title('Collisions', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Collisions')
    axes[1, 0].grid(True, alpha=0.3)

    # Success Rate
    if success_window > 1:
        success_rate = df['terminated'].rolling(window=success_window).mean() * 100
        axes[1, 1].plot(df['episode'], success_rate, linewidth=2, color='purple')
        axes[1, 1].axhline(y=100, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Success Rate ({success_window}-ep window)', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'dashboard_{timestamp}.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {save_dir}/")
    print(f"   - reward_progress_{timestamp}.png")
    print(f"   - coverage_progress_{timestamp}.png")
    print(f"   - efficiency_progress_{timestamp}.png")
    print(f"   - collisions_progress_{timestamp}.png")
    print(f"   - success_rate_{timestamp}.png")
    print(f"   - dashboard_{timestamp}.png")

    # Summary statistics
    print(f"\nTraining Summary Statistics:")
    print(f"   Total Episodes: {len(df)}")
    print(f"   Successful Episodes: {df['terminated'].sum()} ({df['terminated'].mean() * 100:.1f}%)")
    print(f"   Average Reward: {df['total_reward'].mean():.2f}")
    print(f"   Average Coverage: {df['coverage_pct'].mean() * 100:.1f}%")
    print(f"   Average Collisions: {df['total_collisions'].mean():.2f}")
    if len(successful) > 0:
        print(f"   Average Steps to Complete: {successful['steps'].mean():.1f}")


# =======================
# Training & Evaluation
# =======================
import argparse

if __name__ == "__main__":
    # ========================================
    # CLI ARGUMENT PARSING
    # ========================================
    parser = argparse.ArgumentParser(description="Multi-Roomba PPO Training i Vizualizacija")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--treniraj-novi", action="store_true",
                            help="Treniraj novi model od pocetka")
    mode_group.add_argument("--ucitaj-model", type=str, metavar="PATH",
                            help="Ucitaj postojeci model (npr: models/ppo_multi_roomba_final.zip)")
    mode_group.add_argument("--ucitaj-best", action="store_true",
                            help="Ucitaj najbolji sacuvani model")

    # Visualization flag
    parser.add_argument("--viz", action="store_true",
                        help="Prikazi vizualizaciju nakon treninga/ucitavanja")
    parser.add_argument("--viz-epizode", type=int, default=3,
                        help="Broj epizoda za vizualizaciju (default: 3)")

    args = parser.parse_args()

    # ========================================
    # SETUP
    # ========================================
    print("=" * 60)
    print("Multi-Roomba Ciscenje - PPO Training")
    print("=" * 60)
    print(f"Device: {MODEL_DEVICE.upper()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Grid: {GRID_SIZE}×{GRID_SIZE} | Roombas: {N_ROOMBAS}")
    print("=" * 60)

    # Create directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    model = None

    # ========================================
    # TRAINING MODE
    # ========================================
    if args.treniraj_novi:
        print(f"\nTotal timesteps: {TOTAL_TIMESTEPS:,}")
        print("=" * 60)

        # Training environment
        train_env = MultiRoombaGrid(
            grid_size=GRID_SIZE,
            n_roombas=N_ROOMBAS,
            render_mode=None,
            max_steps=500,
            verbose=False
        )

        # Evaluation environment
        eval_env = MultiRoombaGrid(
            grid_size=GRID_SIZE,
            n_roombas=N_ROOMBAS,
            render_mode=None,
            max_steps=500,
            verbose=True,
            log_dir=os.path.join(LOG_DIR, "eval")
        )

        # PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95, # Generalized Advantage Estimation
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device=MODEL_DEVICE,
            tensorboard_log=TENSORBOARD_LOG
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=SAVE_PATH,
            name_prefix="ppo_multi_roomba",
            verbose=1
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(SAVE_PATH, "best_model"),
            log_path=os.path.join(LOG_DIR, "eval"),
            eval_freq=2500,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )

        # Training
        print("\nPokrenut training...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        # Save final model
        final_path = os.path.join(SAVE_PATH, "ppo_multi_roomba_final")
        model.save(final_path)
        print(f"\nTraining complete! Model saved to: {final_path}.zip")

        # Generate Training Plots
        print("\nGenerisanje training plot-ova...")
        csv_path = os.path.join(LOG_DIR, "episode_data.csv")
        if os.path.exists(csv_path):
            plot_training_metrics(csv_path, PLOT_DIR)
        else:
            print(f"No training data found at {csv_path}")

    # ========================================
    # LOAD EXISTING MODEL
    # ========================================
    elif args.ucitaj_model:
        model_path = args.ucitaj_model

        # Add .zip if not present
        if not model_path.endswith(".zip"):
            model_path += ".zip"

        print(f"\nUcitavam model: {model_path}")

        if not os.path.exists(model_path):
            print(f"GRESKA: Model ne postoji na putanji: {model_path}")
            exit(1)

        try:
            model = PPO.load(model_path, device=MODEL_DEVICE)
            print("Model uspesno ucitan!")
        except Exception as e:
            print(f"GRESKA pri ucitavanju modela: {e}")
            exit(1)

    # ========================================
    # LOAD BEST MODEL
    # ========================================
    elif args.ucitaj_best:
        best_model_path = os.path.join(SAVE_PATH, "best_model", "best_model.zip")

        print(f"\nUcitavam najbolji model: {best_model_path}")

        if not os.path.exists(best_model_path):
            print(f"GRESKA: Best model ne postoji na putanji: {best_model_path}")
            print("Molim vas prvo trenirajte model sa --treniraj-novi")
            exit(1)

        try:
            model = PPO.load(best_model_path, device=MODEL_DEVICE)
            print("Best model uspesno ucitan!")
        except Exception as e:
            print(f"GRESKA pri ucitavanju modela: {e}")
            exit(1)

    # ========================================
    # VISUALIZATION
    # ========================================
    if args.viz:
        if model is None:
            print("\nGRESKA: Nema modela za vizualizaciju!")
            exit(1)

        episodes = args.viz_epizode
        print(f"\nPokrecem {episodes} vizualizacija epizoda...")

        viz_env = MultiRoombaGrid(
            grid_size=GRID_SIZE,
            n_roombas=N_ROOMBAS,
            render_mode="human",
            max_steps=500,
            verbose=True,
            log_dir=None
        )

        for ep in range(episodes):
            print(f"\nEpizoda {ep + 1}/{episodes}")
            obs, _ = viz_env.reset()
            done = False
            step_count = 0

            while not done:
                actions, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = viz_env.step(actions)
                done = terminated or truncated
                step_count += 1

            print(f"Epizoda zavrsena nakon {step_count} koraka")
            pygame.time.wait(1000)

        viz_env.close()

    print("\nSve gotovo!")