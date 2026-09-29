"""Run the random baseline and save numerical and plot results."""

import csv
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv  # registers the environment
from myagent import RandomAgent


# this is our environment ID registered with Gymnasium.
ENV_ID = "cs272/GreenHouse-v0"

# directories used to save the random baseline results and episode plots.
OUT = Path("results/random_baseline")
PLOT_OUT = OUT / "episode_plots"

# these seed values are used to run the random agent multiple times.
SEEDS = [0, 1, 2, 3, 4]

# number of episodes used to run the random agent.
EPISODES = 200

# window used for moving average return calculation.
WINDOW = 100

# seed used to save sample episode plots.
PLOT_SEED = 0

# selected episodes are plotted every 25 episodes.
CHECKPOINTS = range(25, EPISODES + 1, 25)


# wrapper used to keep track of states, episode histories, and premature harvests.
class EpisodeTracker(gym.Wrapper):
    """Record states, final states, and premature harvests."""

    def __init__(self, env):
        super().__init__(env)

        # store environment information and episode data.
        self.info = {}
        self.state = None
        self.current_episode = []
        self.episodes = []
        self.premature_harvests = 0
        self.episode_stats = []

    def reset(self, **kwargs):
        # reset the environment and store the starting state information.
        observation, self.info = self.env.reset(**kwargs)
        self.state = int(observation)

        # clear the data from the previous episode.
        self.current_episode = []
        self.premature_harvests = 0

        return observation, self.info

    def step(self, action):
        # save the growth stage and moisture level before taking the action.
        state = (
            self.info["growth_stage"],
            self.info["moisture"],
        )

        # count harvest actions taken before the plant reaches the ripe stage.
        if action == 2 and self.info["growth_stage"] < 5:
            self.premature_harvests += 1

        # take the random action and observe the result.
        observation, reward, terminated, truncated, self.info = self.env.step(action)
        self.state = int(observation)

        # store the state, action, and reward for the current episode.
        self.current_episode.append((state, int(action), float(reward)))

        # save the episode history and statistics when the episode ends.
        if terminated or truncated:
            self.episodes.append(self.current_episode)
            self.episode_stats.append({
                "last_state": (
                    self.info["growth_stage"],
                    self.info["moisture"],
                ),
                "premature_harvests": self.premature_harvests,
            })

        return observation, reward, terminated, truncated, self.info


# function for calculating the moving average of the returns obtained by the random agent.
def moving_average(values, window=WINDOW):
    return np.convolve(values, np.ones(window) / window, mode="valid")


# function for saving the results into a csv file.
def save_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# this function plots the rewards and states from one random agent episode.
def plot_episode(history, episode_number, seed):
    """Plot reward at every timestep and annotate its (growth, moisture) state."""

    # create the timestep values and rewards for the plot.
    steps = np.arange(1, len(history) + 1)
    rewards = [item[2] for item in history]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, rewards, marker="o")

    # label each timestep with its growth stage and moisture level.
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
        f"Random agent: episode {episode_number}, seed {seed}\n"
        "Labels show (growth stage, moisture) before each action"
    )
    plt.tight_layout()

    # save the episode plot using its seed and episode number.
    plt.savefig(
        PLOT_OUT / f"random_seed{seed}_episode{episode_number:03d}.png",
        dpi=200,
    )
    plt.close()


# the main function runs the random baseline for all five seeds and saves the results.
def main():
    # create the directories used to store the results and plots.
    OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)

    # store the returns, episode results, summary results, and checkpoint episodes.
    all_returns = []
    episode_rows = []
    summary_rows = []
    checkpoint_histories = {}

    # run the random agent separately for each seed.
    for seed in SEEDS:
        env = EpisodeTracker(gym.make(ENV_ID))
        env.reset(seed=seed)

        # create and train the random agent for the current seed.
        agent = RandomAgent(env=env, total_epi=EPISODES, seed=seed)
        returns = np.asarray(agent.learn(), dtype=float)

        # save the returns obtained from the current seed.
        all_returns.append(returns)

        # save the results from each individual episode.
        for episode, (value, info) in enumerate(
            zip(returns, env.episode_stats), 1
        ):
            episode_rows.append({
                "lambda": "random",
                "seed": seed,
                "episode": episode,
                "last_state": info["last_state"],
                "return": value,
                "premature_harvests": info["premature_harvests"],
            })

            print(
                f"lambda=random seed={seed} episode={episode:3d} "
                f"last_state={info['last_state']} return={value:8.2f} "
                f"premature_harvests={info['premature_harvests']}"
            )

            # save selected episode histories from the plot seed.
            if seed == PLOT_SEED and episode in CHECKPOINTS:
                checkpoint_histories[episode] = list(env.episodes[episode - 1])

        # calculate the mean final return for the current seed.
        final_mean = float(np.mean(returns[-WINDOW:]))

        # save the summary results for the current seed.
        summary_rows.append({
            "lambda": "random",
            "seed": seed,
            "mean_final_100": final_mean,
            "maximum_return": float(np.max(returns)),
        })

        env.close()

    # save the results from every random agent episode.
    save_csv(
        OUT / "random_episode_returns.csv",
        [
            "lambda", "seed", "episode", "last_state", "return",
            "premature_harvests",
        ],
        episode_rows,
    )

    # save the summary results for each seed.
    save_csv(
        OUT / "random_seed_summary.csv",
        ["lambda", "seed", "mean_final_100", "maximum_return"],
        summary_rows,
    )

    # combine the returns from all five seeds and smooth each seed separately.
    matrix = np.asarray(all_returns)
    smoothed = np.asarray([moving_average(row) for row in matrix])
    episodes = np.arange(WINDOW, EPISODES + 1)

    # calculate the mean and standard deviation across all five seeds.
    plt.figure(figsize=(10, 6))
    mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)

    # plot the random baseline learning curve and spread across seeds.
    plt.plot(episodes, mean, label="random baseline")
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel(f"Return ({WINDOW}-episode moving average)")
    plt.title("Random-agent baseline")
    plt.legend()
    plt.tight_layout()

    # save the random baseline learning curve.
    plt.savefig(OUT / "random_learning_curve.png", dpi=200)
    plt.close()

    # create plots for the selected checkpoint episodes.
    for episode, history in checkpoint_histories.items():
        plot_episode(history, episode, PLOT_SEED)

    # print the overall random baseline results across all five seeds.
    print("\nOverall random baseline:")
    print(f"Mean final return: {np.mean(matrix[:, -WINDOW:]):.3f}")
    print(f"Standard deviation: {np.std(matrix[:, -WINDOW:]):.3f}")
    print(f"Learning curve saved in: {OUT / 'random_learning_curve.png'}")
    print(f"Episode plots saved in: {PLOT_OUT}")


if __name__ == "__main__":
    main()