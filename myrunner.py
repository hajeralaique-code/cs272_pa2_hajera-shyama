import gymnasium as gym
import matplotlib.pyplot as plt
import myenv

from myagent import SarsaLambdaAgent


# -------------------------------------------------
# Wrapper that records state, action, and reward
# for every step in every training episode
# -------------------------------------------------
class RewardRecorder(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        # Each item will be one whole episode.
        # Each episode contains:
        # [(state, action, reward), ...]
        self.episodes = []

        self.current_state = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Start a new episode recording
        self.episodes.append([])

        # Save the starting state
        self.current_state = observation

        return observation, info

    def step(self, action):

        # State BEFORE the action is taken
        state = self.current_state

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Save:
        # current state
        # chosen action
        # reward received
        self.episodes[-1].append(
            (state, action, reward)
        )

        # Update current state for the next step
        self.current_state = observation

        return observation, reward, terminated, truncated, info


# -------------------------------------------------
# Create environment
# -------------------------------------------------
base_env = gym.make("cs272/GreenHouse-v0")

env = RewardRecorder(base_env)


# -------------------------------------------------
# Create SARSA(lambda) agent
# -------------------------------------------------
agent = SarsaLambdaAgent(
    env=env,
    total_epi=200,
    lam=0.9,
    seed=42,
)


# -------------------------------------------------
# Train
# -------------------------------------------------
print("Training agent...")

returns = agent.learn()

print("Training finished!")


# -------------------------------------------------
# Episodes we want to examine
# -------------------------------------------------
checkpoints = [1, 50, 100, 175, 200]


# Action number -> readable action name
action_names = {
    0: "Water",
    1: "Wait",
    2: "Harvest"
}


# -------------------------------------------------
# Plot selected training episodes
# -------------------------------------------------
for episode_number in checkpoints:

    # Python lists begin at index 0,
    # so episode 1 is stored at index 0
    episode = env.episodes[episode_number - 1]

    states = [
        state
        for state, _, _ in episode
    ]

    actions = [
        action
        for _, action, _ in episode
    ]

    rewards = [
        reward
        for _, _, reward in episode
    ]

    steps = range(1, len(episode) + 1)


    # -------------------------------------------------
    # Create reward plot
    # -------------------------------------------------
    plt.figure(figsize=(11, 6))

    plt.plot(
        steps,
        rewards,
        marker="o"
    )


    # -------------------------------------------------
    # Add state + action label to every point
    # -------------------------------------------------
    for step, state, action, reward in zip(
        steps,
        states,
        actions,
        rewards
    ):

        # Decode greenhouse state
        growth = state // 5
        moisture = state % 5

        label = (
            f"State {state}\n"
            f"G={growth}, M={moisture}\n"
            f"{action_names[action]}"
        )

        plt.annotate(
            label,
            (step, reward),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8
        )


    # -------------------------------------------------
    # Graph labels
    # -------------------------------------------------
    plt.xlabel("Step")

    plt.ylabel("Reward")

    plt.title(
        f"Training Episode {episode_number}: "
        f"State, Action, and Reward"
    )

    plt.tight_layout()

    plt.show()


    # -------------------------------------------------
    # Print detailed episode information
    # -------------------------------------------------
    print("\n-----------------------------------")
    print(f"Episode {episode_number}")
    print("-----------------------------------")

    for step, (state, action, reward) in enumerate(
        episode,
        start=1
    ):

        growth = state // 5
        moisture = state % 5

        print(
            f"Step {step}: "
            f"State {state} "
            f"(growth={growth}, moisture={moisture}) | "
            f"Action={action_names[action]} | "
            f"Reward={reward}"
        )


    total_return = sum(
        reward
        for _, _, reward in episode
    )

    print("Total return:", total_return)


# -------------------------------------------------
# Basic training information
# -------------------------------------------------
print("\n===================================")
print("Training Summary")
print("===================================")

print("Number of episodes:", len(returns))

print("Q-table shape:", agent.q.shape)

print("First 10 returns:")
print(returns[:10])

print("Last 10 returns:")
print(returns[-10:])


# -------------------------------------------------
# Close environment
# -------------------------------------------------
env.close()