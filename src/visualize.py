"""
Standalone visualizer for trained MARL agents.

Loads saved model weights (read-only) and renders episodes via pygame.
Optionally captures frames to GIF for presentation use.

Usage examples:
    # Live view of PPO seed 42 (final weights)
    python src/visualize.py --algo ppo --seed 42

    # Save a GIF of random baseline
    python src/visualize.py --algo random --gif gifs/random_demo.gif

    # PPO at specific training checkpoint
    python src/visualize.py --algo ppo --seed 42 --step 100000

    # IDQN showing non-stationarity collapse
    python src/visualize.py --algo idqn --seed 42 --gif gifs/idqn_collapse.gif

    # Side-by-side comparison (run separately, then combine GIFs)
    python src/visualize.py --algo ppo --seed 42 --episodes 3 --gif gifs/ppo.gif
    python src/visualize.py --algo random --episodes 3 --gif gifs/random.gif

    # PPO training progression across checkpoints
    python src/visualize.py --algo ppo --seed 42 --step 10000 --episodes 1 --gif gifs/ppo_10k.gif
    python src/visualize.py --algo ppo --seed 42 --step 100000 --episodes 1 --gif gifs/ppo_100k.gif
    python src/visualize.py --algo ppo --seed 42 --step 1000000 --episodes 1 --gif gifs/ppo_1m.gif

    # Slower playback for screenshots
    python src/visualize.py --algo ppo --seed 42 --fps 5

    # Faster for quick preview
    python src/visualize.py --algo mappo --seed 42 --fps 30
"""

import os
import sys
import argparse
import numpy as np
import torch
import pygame
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from roomba_env import MultiRoombaGrid, GRID_SIZE, N_ROOMBAS
from algos.common import DQNNetwork, set_all_seeds
from algos.mappo import MAPPOActor


# -------------------------------------------------------
# Model loaders — each returns a policy_fn(obs) -> actions
# -------------------------------------------------------

def load_ppo(model_dir, seed, step=None):
    from stable_baselines3 import PPO

    if step is not None:
        path = os.path.join(model_dir, f"ppo/seed_{seed}/models/ppo_{step}_steps")
    else:
        path = os.path.join(model_dir, f"ppo/seed_{seed}/models/ppo_1000000_steps")

    if not os.path.exists(path + ".zip"):
        # Try best model
        path = os.path.join(model_dir, f"ppo/seed_{seed}/models/best/best_model")

    model = PPO.load(path, device="cpu")

    def policy_fn(obs):
        actions, _ = model.predict(obs, deterministic=True)
        return actions

    label = f"PPO (seed {seed}"
    if step is not None:
        label += f", {step:,} steps"
    label += ")"
    return policy_fn, label


def load_mappo(model_dir, seed):
    n_agents = N_ROOMBAS
    grid_size = GRID_SIZE
    base_obs_dim = grid_size ** 2 + 2 * n_agents
    actor_obs_dim = base_obs_dim + n_agents
    device = torch.device("cpu")

    actor = MAPPOActor(actor_obs_dim, 5).to(device)
    weight_path = os.path.join(model_dir, f"mappo/seed_{seed}/models/mappo_actor.pt")
    actor.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
    actor.eval()

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

    return policy_fn, f"MAPPO (seed {seed})"


def load_idqn(model_dir, seed):
    n_agents = N_ROOMBAS
    obs_dim = GRID_SIZE ** 2 + 2 * n_agents
    device = torch.device("cpu")

    dqns = [DQNNetwork(obs_dim, 5).to(device) for _ in range(n_agents)]
    for i, dqn in enumerate(dqns):
        weight_path = os.path.join(model_dir, f"idqn/seed_{seed}/models/idqn_agent_{i}.pt")
        dqn.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
        dqn.eval()

    def policy_fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        actions = []
        for dqn in dqns:
            with torch.no_grad():
                q = dqn(obs_t)
                actions.append(q.argmax(dim=1).item())
        return np.array(actions)

    return policy_fn, f"IDQN (seed {seed})"


def load_qmix(model_dir, seed):
    n_agents = N_ROOMBAS
    obs_dim = GRID_SIZE ** 2 + 2 * n_agents
    device = torch.device("cpu")

    agent_nets = [DQNNetwork(obs_dim, 5).to(device) for _ in range(n_agents)]
    for i, net in enumerate(agent_nets):
        weight_path = os.path.join(model_dir, f"qmix/seed_{seed}/models/qmix_agent_{i}.pt")
        net.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
        net.eval()

    def policy_fn(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        actions = []
        for net in agent_nets:
            with torch.no_grad():
                q = net(obs_t)
                actions.append(q.argmax(dim=1).item())
        return np.array(actions)

    return policy_fn, f"QMIX (seed {seed})"


def load_dqn_joint(model_dir, seed):
    from stable_baselines3 import DQN

    path = os.path.join(model_dir, f"dqn_joint/seed_{seed}/dqn_joint_final")
    model = DQN.load(path, device="cpu")

    n_agents = N_ROOMBAS
    actions_per_agent = 5

    def policy_fn(obs):
        flat_action, _ = model.predict(obs, deterministic=True)
        actions = []
        remaining = int(flat_action)
        for _ in range(n_agents):
            actions.append(remaining % actions_per_agent)
            remaining //= actions_per_agent
        return np.array(actions)

    return policy_fn, f"DQN Joint (seed {seed})"


def load_random(seed):
    rng = np.random.RandomState(seed)

    def policy_fn(obs):
        return rng.randint(0, 5, size=N_ROOMBAS)

    return policy_fn, f"Random (seed {seed})"


LOADERS = {
    "ppo": load_ppo,
    "mappo": load_mappo,
    "idqn": load_idqn,
    "qmix": load_qmix,
    "dqn": load_dqn_joint,
    "dqn_joint": load_dqn_joint,
    "random": load_random,
}


# -------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------

COLORS = {
    "bg": (40, 40, 40),
    "clean": (220, 240, 220),
    "dirty": (25, 25, 25),
    "grid_line": (60, 60, 60),
    "agent_normal": (50, 120, 255),
    "agent_collision": (255, 50, 50),
    "agent_outline": (255, 255, 255),
    "text": (255, 255, 255),
    "text_shadow": (0, 0, 0),
    "hud_bg": (0, 0, 0, 160),
}

CELL_SIZE = 80
HUD_HEIGHT = 48


def render_frame(screen, env, label, ep, step, coverage_pct, reward):
    """Draw the current environment state with HUD overlay."""
    grid_px = env.grid_size * CELL_SIZE

    screen.fill(COLORS["bg"])

    # Draw grid tiles
    for y in range(env.grid_size):
        for x in range(env.grid_size):
            color = COLORS["clean"] if env.coverage[y, x] else COLORS["dirty"]
            rect = (x * CELL_SIZE, HUD_HEIGHT + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, COLORS["grid_line"], rect, 1)

    # Draw agents
    position_counts = {}
    for pos in env.positions:
        key = tuple(pos)
        position_counts[key] = position_counts.get(key, 0) + 1

    agent_colors = [(50, 120, 255), (255, 160, 50)]  # blue, orange
    for idx, (x, y) in enumerate(env.positions):
        colliding = position_counts[(x, y)] > 1
        color = COLORS["agent_collision"] if colliding else agent_colors[idx % len(agent_colors)]
        center = (x * CELL_SIZE + CELL_SIZE // 2,
                  HUD_HEIGHT + y * CELL_SIZE + CELL_SIZE // 2)
        radius = CELL_SIZE // 3
        pygame.draw.circle(screen, color, center, radius)
        pygame.draw.circle(screen, COLORS["agent_outline"], center, radius, 2)

        # Agent label
        font_sm = pygame.font.SysFont("consolas", 16, bold=True)
        id_surf = font_sm.render(str(idx), True, COLORS["text"])
        id_rect = id_surf.get_rect(center=center)
        screen.blit(id_surf, id_rect)

    # HUD bar
    hud_rect = pygame.Surface((grid_px, HUD_HEIGHT), pygame.SRCALPHA)
    hud_rect.fill((0, 0, 0, 180))
    screen.blit(hud_rect, (0, 0))

    font = pygame.font.SysFont("consolas", 16, bold=True)
    left_text = f"{label}  |  Ep {ep}  Step {step}"
    right_text = f"Coverage {coverage_pct:.0f}%  Reward {reward:.1f}"

    left_surf = font.render(left_text, True, COLORS["text"])
    right_surf = font.render(right_text, True, COLORS["text"])

    # Stack vertically to avoid overlap on narrow grids
    screen.blit(left_surf, (8, 6))
    screen.blit(right_surf, (grid_px - right_surf.get_width() - 8, 26))


def surface_to_pil(screen):
    """Capture current pygame surface as a PIL Image."""
    data = pygame.image.tostring(screen, "RGB")
    size = screen.get_size()
    return Image.frombytes("RGB", size, data)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def run(args):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    results_dir = os.path.normpath(results_dir)

    # Load policy
    algo = args.algo.lower()
    if algo not in LOADERS:
        print(f"Unknown algorithm: {algo}")
        print(f"Available: {', '.join(LOADERS.keys())}")
        sys.exit(1)

    print(f"Loading {algo} (seed {args.seed})...")
    if algo == "random":
        policy_fn, label = load_random(args.seed)
    elif algo == "ppo":
        policy_fn, label = load_ppo(results_dir, args.seed, step=args.step)
    else:
        loader = LOADERS[algo]
        policy_fn, label = loader(results_dir, args.seed)

    print(f"  -> {label}")

    # Create environment (no logging — read-only session)
    set_all_seeds(args.seed)
    env = MultiRoombaGrid(
        grid_size=GRID_SIZE,
        n_roombas=N_ROOMBAS,
        render_mode=None,  # we handle rendering ourselves
        max_steps=500,
        verbose=False,
        log_dir=None,
    )

    # Init pygame display
    grid_px = GRID_SIZE * CELL_SIZE
    window_h = grid_px + HUD_HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((grid_px, window_h))
    pygame.display.set_caption(f"MARL Visualizer — {label}")
    clock = pygame.time.Clock()

    frames = []  # for GIF capture
    running = True

    for ep in range(1, args.episodes + 1):
        if not running:
            break

        obs, _ = env.reset()
        done = False
        step_count = 0
        ep_reward = 0.0

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break

            if not running:
                break

            actions = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(actions)
            ep_reward += reward
            step_count += 1
            done = terminated or truncated

            coverage_pct = env.coverage.sum() / (env.grid_size ** 2) * 100

            render_frame(screen, env, label, ep, step_count, coverage_pct, ep_reward)
            pygame.display.flip()
            clock.tick(args.fps)

            if args.gif:
                frames.append(surface_to_pil(screen))

        if running:
            status = "COMPLETE" if terminated else "TRUNCATED"
            coverage_pct = env.coverage.sum() / (env.grid_size ** 2) * 100
            print(f"  Episode {ep}: {status} | {step_count} steps | "
                  f"coverage {coverage_pct:.1f}% | reward {ep_reward:.1f}")

            # Pause briefly between episodes
            if ep < args.episodes:
                pygame.time.wait(500)

    # Save GIF
    if args.gif and frames:
        gif_dir = os.path.dirname(args.gif)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)

        # Calculate frame duration from FPS
        duration_ms = max(1000 // args.fps, 20)

        frames[0].save(
            args.gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        size_mb = os.path.getsize(args.gif) / (1024 * 1024)
        print(f"\n  GIF saved: {args.gif} ({len(frames)} frames, {size_mb:.1f} MB)")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize trained MARL agents on the Roomba grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/visualize.py --algo ppo --seed 42
  python src/visualize.py --algo random --gif gifs/random.gif
  python src/visualize.py --algo ppo --seed 42 --step 10000 --gif gifs/ppo_early.gif
  python src/visualize.py --algo idqn --seed 42 --fps 5
        """,
    )
    parser.add_argument("--algo", required=True,
                        choices=["ppo", "mappo", "idqn", "qmix", "dqn", "dqn_joint", "random"],
                        help="Algorithm to visualize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed / model seed (default: 42)")
    parser.add_argument("--step", type=int, default=None,
                        help="PPO checkpoint step (e.g. 10000, 100000). Default: final (1M)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run (default: 3)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Playback speed in frames per second (default: 10)")
    parser.add_argument("--gif", type=str, default=None,
                        help="Path to save GIF (e.g. gifs/demo.gif). Omit for live view only")

    run(parser.parse_args())
