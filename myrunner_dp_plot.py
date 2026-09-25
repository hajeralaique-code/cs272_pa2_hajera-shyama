import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv
import mydp
from myagent import SarsaLambdaAgent


ENV_ID = "cs272/GreenHouse-v0"

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]

EPISODES = 1000
CHUNK = 50  # train in chunks; take one greedy checkpoint after each chunk
CHECKPOINT_EPISODES = np.arange(CHUNK, EPISODES + 1, CHUNK)
WINDOW = 100
TARGET_RETURN = 40.0

GAMMA = 0.99
ALPHA = 0.05
EPSILON = 0.10
INIT_VAL = 1.0

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def train_one(lam, seed):
    """Train one agent in chunks, taking a greedy (exploration-off) checkpoint
    after each chunk so we can see the actual learned policy's return,
    unpolluted by the epsilon exploration that never turns off in learn()."""
    env = gym.make(ENV_ID)
    agent = SarsaLambdaAgent(
        env=env,
        gamma=GAMMA,
        alpha=ALPHA,
        eps=EPSILON,
        lam=lam,
        trace="accumulating",
        total_epi=CHUNK,
        init_val=INIT_VAL,
        seed=seed,
    )

    training_returns, greedy_returns = [], []

    for _ in range(EPISODES // CHUNK):
        training_returns.extend(agent.learn())

        episode, _ = agent.best_run()
        greedy_returns.append(agent.calc_return(episode))

        # Advance the seed base so the next chunk's per-episode seeds don't
        # repeat the ones this chunk just used.
        if agent.seed is not None:
            agent.seed += CHUNK

    env.close()
    return np.asarray(training_returns, dtype=float), np.asarray(greedy_returns, dtype=float)


def run_lambda_sweep():
    all_training_returns = {}
    all_greedy_returns = {}

    print("\n===================================")
    print("SARSA(lambda) SWEEP")
    print("===================================")

    for lam in LAMBDAS:
        all_training_returns[lam] = []
        all_greedy_returns[lam] = []

        print(f"\nLambda = {lam}")

        for seed in SEEDS:
            print(f"  Training seed {seed}...")

            training_returns, greedy_returns = train_one(lam, seed)
            all_training_returns[lam].append(training_returns)
            all_greedy_returns[lam].append(greedy_returns)

    return all_training_returns, all_greedy_returns


def plot_learning_curves(all_greedy_returns, optimal_return, dp_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.plot(range(1, len(dp_history) + 1), dp_history, color="black")
    ax1.axhline(optimal_return, color="black", linestyle="--", alpha=0.4)
    ax1.set_xlabel("Value-iteration sweep")
    ax1.set_ylabel("Return")
    ax1.set_title("Dynamic programming convergence")

    for lam, color in zip(LAMBDAS, COLORS):
        matrix = np.asarray(all_greedy_returns[lam])  # (seeds, n_checkpoints)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)

        ax2.plot(
            CHECKPOINT_EPISODES, mean,
            color=color, linewidth=2, marker="o", markersize=3,
            label=f"lambda={lam}",
        )
        ax2.fill_between(CHECKPOINT_EPISODES, mean - std, mean + std, color=color, alpha=0.12)

    ax2.axhline(optimal_return, color="black", linestyle="--", label=f"DP optimal = {optimal_return:.1f}")
    ax2.set_xlabel("Episode")
    ax2.set_title("SARSA(lambda): greedy policy return (exploration off)")
    ax2.legend()

    fig.suptitle("SARSA(lambda) vs. dynamic programming")
    plt.tight_layout()
    plt.savefig("sarsa_lambda_learning_curves.png", dpi=200)
    plt.close()


def print_summary_table(all_training_returns):
    print("\n===================================")
    print("LAMBDA SUMMARY")
    print("===================================")

    print(f"Target return = {TARGET_RETURN}")
    print(f"Moving-average window = {WINDOW}")
    print()
    print(f"{'lambda':<8}{'episodes to target':<22}{'mean final return':<20}")
    print("-" * 50)

    for lam in LAMBDAS:
        matrix = np.asarray(all_training_returns[lam])
        smoothed = np.asarray([moving_average(row) for row in matrix])
        mean_curve = smoothed.mean(axis=0)

        hits = np.where(mean_curve >= TARGET_RETURN)[0]
        target_text = "not reached" if len(hits) == 0 else str(int(hits[0] + WINDOW))

        mean_final = np.mean(matrix[:, -WINDOW:])

        print(f"{lam:<8}{target_text:<22}{mean_final:<20.3f}")


def print_greedy_episode():
    print("\n===================================")
    print("TRAINED GREEDY SAMPLE EPISODE")
    print("===================================")

    env = gym.make(ENV_ID, render_mode="ansi")
    agent = SarsaLambdaAgent(
        env=env, gamma=GAMMA, alpha=ALPHA, eps=EPSILON,
        lam=0.3, trace="accumulating", total_epi=EPISODES,
        init_val=INIT_VAL, seed=0,
    )
    agent.learn()

    state, info = env.reset(seed=0)
    total_return = 0.0
    print(env.render())

    for step in range(1, 301):
        action = agent.eps_greedy(state, exploration=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_return += reward

        action_name = {0: "Water", 1: "Wait", 2: "Harvest"}[action]
        print(f"\nStep {step}: {action_name}, reward={reward}")
        print(env.render())

        if terminated or truncated:
            break
        state = next_state

    print(f"\nTotal greedy return: {total_return}")
    env.close()


def main():
    V, Q, policy, n_iters, dp_history = mydp.value_iteration()
    optimal_return = V[mydp.START_STATE]
    print(f"DP optimal expected return: {optimal_return:.2f} (converged in {n_iters} sweeps)")

    all_training_returns, all_greedy_returns = run_lambda_sweep()
    plot_learning_curves(all_greedy_returns, optimal_return, dp_history)
    print_summary_table(all_training_returns)
    print_greedy_episode()
    print("\nSaved plot: sarsa_lambda_learning_curves.png")


if __name__ == "__main__":
    main()