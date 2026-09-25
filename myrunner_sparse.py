import csv
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv_sparse
from myagent import SarsaLambdaAgent


ENV_ID = "cs272/GreenHouseSparse-v0"

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]

EPISODES = 1000
WINDOW = 100
TARGET_RETURN = 50.0

GAMMA = 0.99
ALPHA = 0.05
EPSILON = 0.10
INIT_VAL = 1.0

PLOT_FILE = "sparse_lambda_learning_curves.png"
SUMMARY_FILE = "sparse_lambda_summary.csv"

EPISODE_PLOT_DIR = "sparse_episode_plots"

PLOT_LAMBDA = 0.9
PLOT_SEED = 0

CHECKPOINTS = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    300,
    500,
    750,
    1000,
]


class EpisodeTracker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.info = {}
        self.episodes = []
        self.current_episode = []

    def reset(self, **kwargs):
        observation, self.info = self.env.reset(**kwargs)

        self.current_episode = []

        return observation, self.info

    def step(self, action):
        state = (
            self.info["growth_stage"],
            self.info["moisture"],
        )

        observation, reward, terminated, truncated, self.info = (
            self.env.step(action)
        )

        self.current_episode.append(
            (
                state,
                int(action),
                float(reward),
            )
        )

        if terminated or truncated:
            self.episodes.append(
                list(self.current_episode)
            )

        return observation, reward, terminated, truncated, self.info


def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def train_one(lam, seed):
    env = EpisodeTracker(
        gym.make(ENV_ID)
    )

    agent = SarsaLambdaAgent(
        env=env,
        gamma=GAMMA,
        alpha=ALPHA,
        eps=EPSILON,
        lam=lam,
        trace="accumulating",
        total_epi=EPISODES,
        init_val=INIT_VAL,
        seed=seed,
    )

    returns = np.asarray(
        agent.learn(),
        dtype=float,
    )

    episodes = env.episodes

    env.close()

    return returns, episodes


def run_lambda_sweep():
    all_returns = {}
    checkpoint_episodes = None

    print("\n===================================")
    print("SPARSE SARSA(lambda) SWEEP")
    print("===================================")

    for lam in LAMBDAS:
        all_returns[lam] = []

        print(f"\nLambda = {lam}")

        for seed in SEEDS:
            print(f"  Training seed {seed}...")

            returns, episodes = train_one(
                lam,
                seed,
            )

            all_returns[lam].append(
                returns
            )

            if (
                lam == PLOT_LAMBDA
                and seed == PLOT_SEED
            ):
                checkpoint_episodes = episodes

    return all_returns, checkpoint_episodes


def plot_learning_curves(all_returns):
    plt.figure(figsize=(10, 6))

    episodes = np.arange(
        WINDOW,
        EPISODES + 1,
    )

    for lam in LAMBDAS:
        matrix = np.asarray(
            all_returns[lam]
        )

        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        plt.plot(
            episodes,
            mean,
            label=f"lambda={lam}",
        )

        plt.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    plt.axhline(
        TARGET_RETURN,
        linestyle="--",
        label=f"target={TARGET_RETURN}",
    )

    plt.xlabel("Episode")

    plt.ylabel(
        f"Return ({WINDOW}-episode moving average)"
    )

    plt.title(
        "Sparse-Reward SARSA(lambda) Learning Curves"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=200,
    )

    plt.close()


def plot_episode(history, episode_number):
    os.makedirs(
        EPISODE_PLOT_DIR,
        exist_ok=True,
    )

    steps = np.arange(
        1,
        len(history) + 1,
    )

    rewards = [
        item[2]
        for item in history
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        steps,
        rewards,
        marker="o",
    )

    for step, (state, action, reward) in enumerate(
        history,
        start=1,
    ):
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
        f"Sparse SARSA(lambda), "
        f"lambda={PLOT_LAMBDA}, "
        f"seed={PLOT_SEED}, "
        f"episode={episode_number}\n"
        "Labels show (growth stage, moisture) before each action"
    )

    plt.tight_layout()

    plt.savefig(
        f"{EPISODE_PLOT_DIR}/episode_{episode_number:04d}.png",
        dpi=200,
    )

    plt.close()


def build_summary(all_returns):
    rows = []

    for lam in LAMBDAS:
        matrix = np.asarray(
            all_returns[lam]
        )

        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        mean_curve = smoothed.mean(axis=0)

        hits = np.where(
            mean_curve >= TARGET_RETURN
        )[0]

        if len(hits) == 0:
            episodes_to_target = "not reached"
        else:
            episodes_to_target = int(
                hits[0] + WINDOW
            )

        mean_final = float(
            np.mean(
                matrix[:, -WINDOW:]
            )
        )

        rows.append({
            "lambda": lam,
            "episodes_to_target": episodes_to_target,
            "mean_final_return": mean_final,
        })

    return rows


def print_summary_table(rows):
    print("\n===================================")
    print("SPARSE LAMBDA SUMMARY")
    print("===================================")

    print(f"Target return = {TARGET_RETURN}")
    print(f"Moving-average window = {WINDOW}")

    print()

    print(
        f"{'lambda':<8}"
        f"{'episodes to target':<22}"
        f"{'mean final return':<20}"
    )

    print("-" * 50)

    for row in rows:
        print(
            f"{row['lambda']:<8}"
            f"{str(row['episodes_to_target']):<22}"
            f"{row['mean_final_return']:<20.3f}"
        )


def save_summary(rows):
    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "lambda",
                "episodes_to_target",
                "mean_final_return",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def print_greedy_episode():
    print("\n===================================")
    print("SPARSE TRAINED GREEDY SAMPLE")
    print("===================================")

    env = gym.make(
        ENV_ID,
        render_mode="ansi",
    )

    agent = SarsaLambdaAgent(
        env=env,
        gamma=GAMMA,
        alpha=ALPHA,
        eps=EPSILON,
        lam=0.3,
        trace="accumulating",
        total_epi=EPISODES,
        init_val=INIT_VAL,
        seed=0,
    )

    agent.learn()

    state, info = env.reset(seed=0)

    total_return = 0.0

    print(env.render())

    for step in range(1, 301):
        action = agent.eps_greedy(
            state,
            exploration=False,
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_return += reward

        action_name = {
            0: "Water",
            1: "Wait",
            2: "Harvest",
        }[action]

        print(
            f"\nStep {step}: "
            f"{action_name}, reward={reward}"
        )

        print(env.render())

        if terminated or truncated:
            break

        state = next_state

    print(
        f"\nTotal greedy return: {total_return}"
    )

    env.close()


def main():
    all_returns, checkpoint_episodes = run_lambda_sweep()

    plot_learning_curves(
        all_returns
    )

    if checkpoint_episodes is not None:
        for episode_number in CHECKPOINTS:
            plot_episode(
                checkpoint_episodes[
                    episode_number - 1
                ],
                episode_number,
            )

    rows = build_summary(
        all_returns
    )

    print_summary_table(
        rows
    )

    save_summary(
        rows
    )

    print_greedy_episode()

    print(
        f"\nSaved learning curve: {PLOT_FILE}"
    )

    print(
        f"Saved summary: {SUMMARY_FILE}"
    )

    print(
        f"Saved episode plots in: {EPISODE_PLOT_DIR}"
    )


if __name__ == "__main__":
    main()