import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv
from myagent import SarsaLambdaAgent


ENV_ID = "cs272/GreenHouse-v0"

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]

EPISODES = 1000
WINDOW = 100
TARGET_RETURN = 40.0

GAMMA = 0.99
ALPHA = 0.05
EPSILON = 0.10
INIT_VAL = 1.0


def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def train_one(lam, seed):
    env = gym.make(ENV_ID)

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

    env.close()

    return returns


def run_lambda_sweep():
    all_returns = {}

    print("\n===================================")
    print("SARSA(lambda) SWEEP")
    print("===================================")

    for lam in LAMBDAS:
        all_returns[lam] = []

        print(f"\nLambda = {lam}")

        for seed in SEEDS:
            print(f"  Training seed {seed}...")

            returns = train_one(
                lam,
                seed,
            )

            all_returns[lam].append(
                returns
            )

    return all_returns


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
        "SARSA(lambda) Learning Curves"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "sarsa_lambda_learning_curves.png",
        dpi=200,
    )

    plt.close()


def print_summary_table(all_returns):
    print("\n===================================")
    print("LAMBDA SUMMARY")
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

    for lam in LAMBDAS:
        matrix = np.asarray(all_returns[lam])

        # Smooth each seed separately
        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        # Mean learning curve across all seeds
        mean_curve = smoothed.mean(axis=0)

        # Find first point where the mean curve reaches target
        hits = np.where(mean_curve >= TARGET_RETURN)[0]

        if len(hits) == 0:
            target_text = "not reached"
        else:
            episode_to_target = int(hits[0] + WINDOW)
            target_text = str(episode_to_target)

        # Mean return over the final 100 episodes,
        # across all five seeds
        mean_final = np.mean(
            matrix[:, -WINDOW:]
        )

        print(
            f"{lam:<8}"
            f"{target_text:<22}"
            f"{mean_final:<20.3f}"
        )

def print_greedy_episode():
    print("\n===================================")
    print("TRAINED GREEDY SAMPLE EPISODE")
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
    all_returns = run_lambda_sweep()

    plot_learning_curves(
        all_returns
    )

    print_summary_table(
        all_returns
    )

    print_greedy_episode()

    print(
        "\nSaved plot: "
        "sarsa_lambda_learning_curves.png"
    )


if __name__ == "__main__":
    main()