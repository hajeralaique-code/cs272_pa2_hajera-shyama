import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv
from myagent import SarsaLambdaAgent
from mydp import value_iteration, START_STATE


ENV_ID = "cs272/GreenHouse-v0"

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]

EPISODES = 3500
EVALUATION_EPISODES = 500
WINDOW = 100
TARGET_RETURN = 40.0

GAMMA = 1.0
ALPHA = 0.05
EPSILON = 0.10
INIT_VAL = 1.0


def moving_average(values):
    return np.convolve(
        values,
        np.ones(WINDOW) / WINDOW,
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

    returns = np.asarray(agent.learn(), dtype=float)
    env.close()

    return returns, agent


def run_lambda_sweep():
    all_returns = {}
    trained_agents = {}

    for lam in LAMBDAS:
        all_returns[lam] = []
        trained_agents[lam] = []

        for seed in SEEDS:
            print(f"Training lambda={lam}, seed={seed}")

            returns, agent = train_one(lam, seed)

            all_returns[lam].append(returns)
            trained_agents[lam].append(agent)

    return all_returns, trained_agents


def plot_learning_curves(all_returns, dp_value):
    episodes = np.arange(WINDOW, EPISODES + 1)

    plt.figure(figsize=(10, 6))

    for lam in LAMBDAS:
        matrix = np.asarray(all_returns[lam])

        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        plt.plot(
            episodes,
            mean,
            linewidth=2,
            label=f"λ={lam}",
        )

        plt.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    plt.axhline(
        dp_value,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"DP optimal value={dp_value:.2f}",
    )

    plt.axhline(
        TARGET_RETURN,
        color="gray",
        linestyle=":",
        label=f"target={TARGET_RETURN}",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Return")
    plt.title("SARSA(λ) Learning Curves with DP Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sarsa_lambda_learning_curves_with_dp.png", dpi=200)
    plt.close()


def print_summary_table(all_returns):
    print("\nSARSA(λ) SUMMARY")
    print(f"Target return: {TARGET_RETURN}")
    print(f"{'lambda':<10}{'episodes to target':<22}{'mean final return':<20}")

    for lam in LAMBDAS:
        matrix = np.asarray(all_returns[lam])

        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        mean_curve = smoothed.mean(axis=0)
        hits = np.where(mean_curve >= TARGET_RETURN)[0]

        if len(hits) == 0:
            episodes_to_target = "not reached"
        else:
            episodes_to_target = int(hits[0] + WINDOW)

        mean_final = np.mean(matrix[:, -WINDOW:])

        print(
            f"{lam:<10}"
            f"{str(episodes_to_target):<22}"
            f"{mean_final:<20.3f}"
        )


def evaluate_sarsa(agent, seed_offset):
    env = gym.make(ENV_ID)
    returns = []

    for episode in range(EVALUATION_EPISODES):
        state, _ = env.reset(seed=seed_offset + episode)
        total_return = 0.0

        while True:
            action = agent.eps_greedy(
                state,
                exploration=False,
            )

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_return += reward

            if terminated or truncated:
                break

            state = next_state

        returns.append(total_return)

    env.close()
    return np.asarray(returns)


def evaluate_dp(policy):
    env = gym.make(ENV_ID)
    returns = []

    for episode in range(EVALUATION_EPISODES):
        state, _ = env.reset(seed=50000 + episode)
        total_return = 0.0

        while True:
            action = int(policy[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_return += reward

            if terminated or truncated:
                break

            state = next_state

        returns.append(total_return)

    env.close()
    return np.asarray(returns)


def plot_greedy_comparison(trained_agents, dp_value, dp_policy):
    labels = []
    means = []
    stds = []

    for lam in LAMBDAS:
        evaluation_returns = []

        for index, agent in enumerate(trained_agents[lam]):
            values = evaluate_sarsa(
                agent,
                seed_offset=10000 + index * 1000,
            )
            evaluation_returns.extend(values)

        evaluation_returns = np.asarray(evaluation_returns)

        labels.append(f"SARSA λ={lam}")
        means.append(np.mean(evaluation_returns))
        stds.append(np.std(evaluation_returns))

    dp_returns = evaluate_dp(dp_policy)

    labels.append("DP")
    means.append(dp_value)
    stds.append(0.0)

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        x[:-1],
        means[:-1],
        yerr=stds[:-1],
        fmt="o",
        capsize=5,
        linewidth=2,
        label="SARSA greedy evaluation",
    )

    plt.axhline(
        dp_value,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"DP optimal value={dp_value:.2f}",
    )

    plt.xticks(x, labels, rotation=25)
    plt.ylabel("Mean greedy evaluation return")
    plt.title("Greedy SARSA(λ) Policies versus Dynamic Programming")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("greedy_sarsa_vs_dp.png", dpi=200)
    plt.close()

    print("\nGreedy policy evaluation")

    for label, mean, std in zip(labels[:-1], means[:-1], stds[:-1]):
        print(f"{label:<15} mean={mean:.3f}, std={std:.3f}")

    print(f"{'DP':<15} exact value={dp_value:.3f}")
    print(f"DP empirical mean={np.mean(dp_returns):.3f}")


def print_sample_episode(agent):
    env = gym.make(
        ENV_ID,
        render_mode="ansi",
    )

    state, _ = env.reset(seed=0)
    total_return = 0.0

    print("\nTRAINED GREEDY SAMPLE EPISODE")
    print(env.render())

    for step in range(1, 301):
        action = agent.eps_greedy(
            state,
            exploration=False,
        )

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_return += reward

        action_name = {
            0: "Water",
            1: "Wait",
            2: "Harvest",
        }[action]

        print(f"\nStep {step}: {action_name}, reward={reward}")
        print(env.render())

        if terminated or truncated:
            break

        state = next_state

    print(f"\nTotal greedy return: {total_return}")
    env.close()


def main():
    dp_v, dp_q, dp_policy, dp_iterations, _ = value_iteration()
    dp_value = float(dp_v[START_STATE])

    print(f"DP converged in {dp_iterations} iterations")
    print(f"DP optimal value from start state: {dp_value:.3f}")

    all_returns, trained_agents = run_lambda_sweep()

    plot_learning_curves(
        all_returns,
        dp_value,
    )

    print_summary_table(all_returns)

    plot_greedy_comparison(
        trained_agents,
        dp_value,
        dp_policy,
    )

    print_sample_episode(
        trained_agents[0.3][0],
    )

    print("\nSaved files:")
    print("sarsa_lambda_learning_curves_with_dp.png")
    print("greedy_sarsa_vs_dp.png")


if __name__ == "__main__":
    main()