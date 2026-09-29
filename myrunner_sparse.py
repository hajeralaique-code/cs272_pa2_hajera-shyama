import csv
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv_sparse
from myagent import SarsaLambdaAgent


# this is our sparse environment ID registered with Gymnasium.
ENV_ID = "cs272/GreenHouseSparse-v0"

# these lambdas and seed values were explored to see how they affect the agent's learning.
LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]

# number of episodes used to train the sparse environment.
EPISODES = 1000

# window used for moving average return calculation.
WINDOW = 100

# target return used to measure how quickly the agent learns.
TARGET_RETURN = 50.0

# sarsa(lambda) hyperparameters.
GAMMA = 0.99
ALPHA = 0.05
EPSILON = 0.10
INIT_VAL = 1.0

# file names used to save the learning curve and summary results.
PLOT_FILE = "sparse_lambda_learning_curves.png"
SUMMARY_FILE = "sparse_lambda_summary.csv"

# directory used to save plots of individual training episodes.
EPISODE_PLOT_DIR = "sparse_episode_plots"

# lambda and seed used to track sample episodes during training.
PLOT_LAMBDA = 0.9
PLOT_SEED = 0

# selected training episodes that will be plotted to show learning progress.
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


# wrapper used to keep track of the states, actions, and rewards from each episode.
class EpisodeTracker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # store environment info and episode histories.
        self.info = {}
        self.episodes = []
        self.current_episode = []

    def reset(self, **kwargs):
        observation, self.info = self.env.reset(**kwargs)

        # clear the current episode when the environment resets.
        self.current_episode = []

        return observation, self.info

    def step(self, action):
        # save the growth stage and moisture level before taking the action.
        state = (
            self.info["growth_stage"],
            self.info["moisture"],
        )

        # take the action and get the next environment information.
        observation, reward, terminated, truncated, self.info = (
            self.env.step(action)
        )

        # store the state, action, and reward for the current episode.
        self.current_episode.append(
            (
                state,
                int(action),
                float(reward),
            )
        )

        # save the completed episode when it terminates or is truncated.
        if terminated or truncated:
            self.episodes.append(
                list(self.current_episode)
            )

        return observation, reward, terminated, truncated, self.info


# function for calculating the moving average of the returns obtained by the agent during each episode.
def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


# each training run uses a different lambda and seed value.
def train_one(lam, seed):
    # wrap the sparse environment so the training episodes can be recorded.
    env = EpisodeTracker(
        gym.make(ENV_ID)
    )

    # train the agent using the same hyperparameters with the given lambda and seed.
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

    # store the returns obtained during training.
    returns = np.asarray(
        agent.learn(),
        dtype=float,
    )

    # save the recorded episode histories.
    episodes = env.episodes

    # close the environment after training.
    env.close()

    return returns, episodes


# created a dictionary to store the returns obtained for each lambda and seed pair.
def run_lambda_sweep():
    all_returns = {}

    # store the episodes from the selected lambda and seed for checkpoint plots.
    checkpoint_episodes = None

    print("\n===================================")
    print("SPARSE SARSA(lambda) SWEEP")
    print("===================================")

    # train the agent using each lambda value.
    for lam in LAMBDAS:
        all_returns[lam] = []

        print(f"\nLambda = {lam}")

        # train each lambda using all five seeds.
        for seed in SEEDS:
            print(f"  Training seed {seed}...")

            returns, episodes = train_one(
                lam,
                seed,
            )

            all_returns[lam].append(
                returns
            )

            # save the episode histories for the selected lambda and seed.
            if (
                lam == PLOT_LAMBDA
                and seed == PLOT_SEED
            ):
                checkpoint_episodes = episodes

    return all_returns, checkpoint_episodes


# the function plots the learning curve for each lambda using the 100 episode moving average returns.
def plot_learning_curves(all_returns):
    plt.figure(figsize=(10, 6))

    # episode numbers used for the x-axis after applying the moving average.
    episodes = np.arange(
        WINDOW,
        EPISODES + 1,
    )

    for lam in LAMBDAS:
        # combine the returns from all seeds for the current lambda.
        matrix = np.asarray(
            all_returns[lam]
        )

        # smooth the returns from each seed separately.
        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        # calculate the mean and standard deviation across all seeds.
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        plt.plot(
            episodes,
            mean,
            label=f"lambda={lam}",
        )

        # show the spread across the five seeds.
        plt.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    # show the target return on the learning curve.
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

    # save the sparse learning curve.
    plt.savefig(
        PLOT_FILE,
        dpi=200,
    )

    plt.close()


# this function plots the rewards and states from one recorded training episode.
def plot_episode(history, episode_number):
    # create the directory for episode plots if it does not already exist.
    os.makedirs(
        EPISODE_PLOT_DIR,
        exist_ok=True,
    )

    # create the timestep values for the x-axis.
    steps = np.arange(
        1,
        len(history) + 1,
    )

    # get the reward received at each timestep.
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

    # label each timestep with its growth stage and moisture level.
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

    # save the plot using its training episode number.
    plt.savefig(
        f"{EPISODE_PLOT_DIR}/episode_{episode_number:04d}.png",
        dpi=200,
    )

    plt.close()


# this function creates the summary results for each lambda value.
def build_summary(all_returns):
    rows = []

    for lam in LAMBDAS:
        # combine the returns from all five seeds.
        matrix = np.asarray(
            all_returns[lam]
        )

        # smooth each seed separately.
        smoothed = np.asarray([
            moving_average(row)
            for row in matrix
        ])

        # calculate the mean learning curve across all seeds.
        mean_curve = smoothed.mean(axis=0)

        # find where the mean learning curve reaches the target return.
        hits = np.where(
            mean_curve >= TARGET_RETURN
        )[0]

        # record whether and when the target return was reached.
        if len(hits) == 0:
            episodes_to_target = "not reached"
        else:
            episodes_to_target = int(
                hits[0] + WINDOW
            )

        # calculate the mean return over the final 100 episodes across all seeds.
        mean_final = float(
            np.mean(
                matrix[:, -WINDOW:]
            )
        )

        # save the summary values for the current lambda.
        rows.append({
            "lambda": lam,
            "episodes_to_target": episodes_to_target,
            "mean_final_return": mean_final,
        })

    return rows


# summary table was created to show the number of episodes the agent took to reach the target return.
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

    # print the summary values for each lambda.
    for row in rows:
        print(
            f"{row['lambda']:<8}"
            f"{str(row['episodes_to_target']):<22}"
            f"{row['mean_final_return']:<20.3f}"
        )


# this function saves the lambda summary results to a csv file.
def save_summary(rows):
    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as file:

        # create the columns for the summary file.
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "lambda",
                "episodes_to_target",
                "mean_final_return",
            ],
        )

        # save the column names and summary rows.
        writer.writeheader()
        writer.writerows(rows)


# finally printing the sample episode with greedy policy after training the sparse agent.
def print_greedy_episode():
    print("\n===================================")
    print("SPARSE TRAINED GREEDY SAMPLE")
    print("===================================")

    # create the sparse environment in ansi render mode.
    env = gym.make(
        ENV_ID,
        render_mode="ansi",
    )

    # train the agent before running the greedy sample episode.
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

    # reset the environment before starting the greedy episode.
    state, info = env.reset(seed=0)

    total_return = 0.0

    # render the state of the environment before taking any action.
    print(env.render())

    # the greedy policy is used to run the episode and render the environment after each action.
    for step in range(1, 301):
        action = agent.eps_greedy(
            state,
            exploration=False,
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_return += reward

        # convert the action number to its greenhouse action name.
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

        # stop the sample episode when it terminates or is truncated.
        if terminated or truncated:
            break

        state = next_state

    print(
        f"\nTotal greedy return: {total_return}"
    )

    env.close()


# the main function was created to run all functions in sequence.
def main():
    all_returns, checkpoint_episodes = run_lambda_sweep()

    # plot the overall sparse learning curves.
    plot_learning_curves(
        all_returns
    )

    # create plots for the selected checkpoint training episodes.
    if checkpoint_episodes is not None:
        for episode_number in CHECKPOINTS:
            plot_episode(
                checkpoint_episodes[
                    episode_number - 1
                ],
                episode_number,
            )

    # build and print the lambda summary table.
    rows = build_summary(
        all_returns
    )

    print_summary_table(
        rows
    )

    # save the summary table to a csv file.
    save_summary(
        rows
    )

    # print one trained greedy sample episode.
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