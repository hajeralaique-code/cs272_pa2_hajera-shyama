"""Run SARSA(lambda) and save numerical and plot results."""

import csv
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv
from myagent import SarsaLambdaAgent


ENV_ID = "cs272/GreenHouse-v0"
RESULTS_DIR = Path("results/sarsa_lambda")
PLOT_DIR = RESULTS_DIR / "episode_plots"
LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]
EPISODES = 1000
WINDOW = 100
PLOT_LAMBDA = 0.9
PLOT_SEED = 0
CHECKPOINTS = range(25, EPISODES + 1, 25)


class EpisodeTracker(gym.Wrapper):
    """Record states, final states, and premature harvests."""

    def __init__(self, env):
        super().__init__(env)
        self.info = {}
        self.episodes = []
        self.stats = []
        self.current_episode = []
        self.premature_harvests = 0

    def reset(self, **kwargs):
        observation, self.info = self.env.reset(**kwargs)
        self.current_episode = []
        self.premature_harvests = 0
        return observation, self.info

    def step(self, action):
        state = (
            self.info["growth_stage"],
            self.info["moisture"],
        )

        if action == 2 and self.info["growth_stage"] < 5:
            self.premature_harvests += 1

        observation, reward, terminated, truncated, self.info = self.env.step(action)
        self.current_episode.append((state, int(action), float(reward)))

        if terminated or truncated:
            self.episodes.append(self.current_episode)
            self.stats.append({
                "last_state": (
                    self.info["growth_stage"],
                    self.info["moisture"],
                ),
                "premature_harvests": self.premature_harvests,
            })

        return observation, reward, terminated, truncated, self.info


def moving_average(values, window=WINDOW):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def train_one(lam, seed):
    env = EpisodeTracker(gym.make(ENV_ID))
    agent = SarsaLambdaAgent(
        env=env,
        gamma=0.99,
        alpha=0.05,
        eps=0.10,
        lam=lam,
        trace="accumulating",
        total_epi=EPISODES,
        init_val=1.0,
        seed=seed,
    )
    returns = np.asarray(agent.learn(), dtype=float)
    stats, episodes = env.stats, env.episodes
    env.close()
    return returns, stats, episodes


def save_csv(path, fields, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_episode(history, episode_number):
    steps = np.arange(1, len(history) + 1)
    rewards = [item[2] for item in history]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, rewards, marker="o")

    for step, (state, action, reward) in enumerate(history, 1):
        plt.annotate(
            f"({state[0]}, {state[1]})",
            (step, reward),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(
        f"SARSA(lambda), lambda={PLOT_LAMBDA}, seed={PLOT_SEED}, "
        f"episode={episode_number}\n"
        "Labels show (growth stage, moisture) before each action"
    )
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"episode_{episode_number:03d}.png",
        dpi=200,
    )
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    all_returns = {}
    episode_rows = []
    seed_rows = []
    lambda_rows = []
    checkpoint_episodes = None

    for lam in LAMBDAS:
        all_returns[lam] = []
        print(f"\n========== lambda = {lam} ==========")

        for seed in SEEDS:
            returns, stats, histories = train_one(lam, seed)
            all_returns[lam].append(returns)

            if lam == PLOT_LAMBDA and seed == PLOT_SEED:
                checkpoint_episodes = histories

            for episode, (value, info) in enumerate(zip(returns, stats), 1):
                episode_rows.append({
                    "lambda": lam,
                    "seed": seed,
                    "episode": episode,
                    "last_state": info["last_state"],
                    "return": value,
                    "premature_harvests": info["premature_harvests"],
                })

                print(
                    f"lambda={lam:<3} seed={seed} episode={episode:3d} "
                    f"last_state={info['last_state']} return={value:8.2f} "
                    f"premature_harvests={info['premature_harvests']}"
                )

            seed_rows.append({
                "lambda": lam,
                "seed": seed,
                "mean_final_100": float(np.mean(returns[-WINDOW:])),
                "max_return": float(np.max(returns)),
            })

        matrix = np.asarray(all_returns[lam])
        lambda_rows.append({
            "lambda": lam,
            "mean_final_100": float(np.mean(matrix[:, -WINDOW:])),
            "std_final_100": float(np.std(matrix[:, -WINDOW:])),
            "mean_max_return": float(np.mean(np.max(matrix, axis=1))),
        })

    save_csv(
        RESULTS_DIR / "sarsa_episode_returns.csv",
        [
            "lambda", "seed", "episode", "last_state", "return",
            "premature_harvests",
        ],
        episode_rows,
    )
    save_csv(
        RESULTS_DIR / "sarsa_seed_summary.csv",
        ["lambda", "seed", "mean_final_100", "max_return"],
        seed_rows,
    )
    save_csv(
        RESULTS_DIR / "sarsa_lambda_summary.csv",
        ["lambda", "mean_final_100", "std_final_100", "mean_max_return"],
        lambda_rows,
    )

    plt.figure(figsize=(10, 6))
    episodes = np.arange(WINDOW, EPISODES + 1)

    for lam in LAMBDAS:
        smoothed = np.asarray([
            moving_average(row) for row in all_returns[lam]
        ])
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        plt.plot(episodes, mean, label=f"lambda={lam}")
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel(f"Return ({WINDOW}-episode moving average)")
    plt.title("SARSA(lambda) learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sarsa_lambda_learning_curves.png", dpi=200)
    plt.close()

    if checkpoint_episodes is not None:
        for episode in CHECKPOINTS:
            plot_episode(checkpoint_episodes[episode - 1], episode)

    print(f"\nResults saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
