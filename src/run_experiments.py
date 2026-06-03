"""
Multi-seed experiment runner for MARL algorithm comparison.

Usage:
    # Run all algorithms with all seeds
    python src/run_experiments.py

    # Run specific algorithm(s)
    python src/run_experiments.py --algo ppo qmix

    # Quick test with fewer timesteps
    python src/run_experiments.py --algo random --seeds 42 --timesteps 1000

    # Incremental: already-completed (algo, seed) pairs are skipped
"""

import os
import sys
import json
import argparse
import importlib
import time

sys.path.insert(0, os.path.dirname(__file__))

ALGORITHMS = ["ppo", "dqn_joint", "idqn", "qmix", "mappo", "random"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_TIMESTEPS = 10_000_000
DEFAULT_EVAL_EPISODES = 100

ALGO_MODULE_MAP = {
    "ppo": "algos.ppo_centralized",
    "dqn_joint": "algos.dqn_joint",
    "idqn": "algos.idqn",
    "qmix": "algos.qmix",
    "mappo": "algos.mappo",
    "random": "algos.random_baseline",
}

DEFAULT_ENV_CONFIG = {
    "grid_size": 8,
    "n_roombas": 2,
    "max_steps": 500,
}


def run_single(algo, seed, timesteps, eval_episodes, env_config, results_root):
    """Run training + evaluation for one (algo, seed) pair."""
    result_dir = os.path.join(results_root, algo, f"seed_{seed}")
    metrics_path = os.path.join(result_dir, "eval_metrics.json")

    if os.path.exists(metrics_path):
        print(f"[SKIP] {algo} seed={seed} — already complete")
        return

    print("=" * 60)
    print(f"[START] Algorithm: {algo} | Seed: {seed} | Timesteps: {timesteps:,}")
    print("=" * 60)

    os.makedirs(result_dir, exist_ok=True)
    module = importlib.import_module(ALGO_MODULE_MAP[algo])

    # Train
    t0 = time.time()
    model_path = module.train(env_config, seed, timesteps, result_dir)
    train_time = time.time() - t0

    # Evaluate
    t1 = time.time()
    metrics = module.evaluate(model_path, env_config, seed, eval_episodes, result_dir)
    eval_time = time.time() - t1

    metrics["train_time_seconds"] = train_time
    metrics["eval_time_seconds"] = eval_time
    metrics["seed"] = seed
    metrics["timesteps"] = timesteps

    # Save final metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] {algo} seed={seed} | "
          f"Success: {metrics['success_rate']:.1f}% | "
          f"Reward: {metrics['avg_reward']:.1f} | "
          f"Coverage: {metrics['avg_coverage']:.1f}% | "
          f"Collisions: {metrics['avg_collisions']:.1f} | "
          f"Train: {train_time:.0f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Multi-seed MARL experiment runner")
    parser.add_argument("--algo", nargs="+", default=ALGORITHMS,
                        choices=ALGORITHMS, help="Algorithms to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help="Training timesteps per run")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="Evaluation episodes per run")
    parser.add_argument("--results-dir", default="results",
                        help="Root directory for results")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--n-roombas", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    env_config = {
        "grid_size": args.grid_size,
        "n_roombas": args.n_roombas,
        "max_steps": args.max_steps,
    }

    total_runs = len(args.algo) * len(args.seeds)
    print(f"Experiment: {len(args.algo)} algorithms x {len(args.seeds)} seeds = {total_runs} runs")
    print(f"Algorithms: {args.algo}")
    print(f"Seeds: {args.seeds}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Environment: {args.grid_size}x{args.grid_size}, {args.n_roombas} roombas")
    print()

    for algo in args.algo:
        for seed in args.seeds:
            run_single(algo, seed, args.timesteps, args.eval_episodes,
                       env_config, args.results_dir)

    print("=" * 60)
    print("All experiments complete!")
    print(f"Results saved to: {args.results_dir}/")
    print(f"Run 'python src/aggregate_results.py --results-dir {args.results_dir}' to generate summary.")


if __name__ == "__main__":
    main()
