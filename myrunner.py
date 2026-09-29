import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import myenv
from myagent import RandomAgent, SarsaLambdaAgent #importing both my agent and uniform random agent classes

#environment ID registered in Gymnasium
ENV_ID = "cs272/GreenHouse-v0"
#the agent ran SARSA λ algorithm using each λ and seed pair
LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4] #five random seed selected

#total number of episodes selected after testing 1000, 2000, 3000 episodes.
#at episode 3500 each SARSA(λ) learning convergs for a few episodes
EPISODES = 3500
WINDOW = 100 #moving average calculation window
TARGET_RETURN = 40.0 #target return was selected just as a reporting threshold
#SARSA(λ) hyparameters
GAMMA = 0.99 #discount factor
ALPHA = 0.05 #learning rate
EPSILON = 0.10 #epsilon value for epsilon greedy
INIT_VAL = 1.0 #initial value assinged to a Q-table


#the function calculated moving average return smoothed over 100 episodes
def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)
    return np.convolve(values,np.ones(window) / window,mode="valid",)

#the function trains new agent for 3500 episodes using a fixed λ and seed pair
#λ and seed pair is changed every new run
def train_one(lam, seed):
    env = gym.make(ENV_ID)
    #accumulating trace is used to increase trace for a state action pair.
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

    return returns

#the function runs a uniform random agent for a random seed and records undiscounted return for each episode
def train_random(seed):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    agent = RandomAgent(
        env=env,
        total_epi=EPISODES,
        seed=seed,
    )

    returns = np.asarray(agent.learn(), dtype=float)
    env.close()

    return returns

#this function runs new agent for each λ,seed pair and returns a dictionary with each episode return.
def run_lambda_sweep():
    all_returns = {}

    print("\n-----------------------------------")
    print("SARSA(λ) SWEEP")
    print("-----------------------------------")

    for lam in LAMBDAS:
        all_returns[lam] = []
        print(f"\nλ = {lam}")
        for seed in SEEDS:
            print(f"Training seed {seed}")
            returns = train_one(lam, seed)
            all_returns[lam].append(returns)

    return all_returns

#this functions runs a uniform random agent with different seed values and returns the episodic return values.
def run_random_baseline():
    baseline_returns = []

    print("\n-----------------------------------")
    print("UNIFORM-RANDOM BASELINE")
    print("-----------------------------------")

    for seed in SEEDS:
        print(f"Training random baseline seed {seed}")
        returns = train_random(seed)
        baseline_returns.append(returns)

    return np.asarray(baseline_returns, dtype=float)

#this function plots the λ sweep learning curves with the x-axis as the number of episodes and y-axis as the return values.
#each learning curve plots the 100-episode moving average returns and the shaded area for each solid line 
#represents the +/- 1 standard deviation from the mean
def plot_learning_curves(all_returns, random_returns):
    plt.figure(figsize=(10, 6))
    episodes = np.arange(WINDOW, EPISODES + 1)

    for lam in LAMBDAS:
        matrix = np.asarray(all_returns[lam])
        smoothed = np.asarray([moving_average(row) for row in matrix])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        plt.plot(episodes, mean, label=f"λ={lam}") #plotting the learning curve for each lambda
        plt.fill_between(episodes,mean - std,mean + std,alpha=0.15) #plotting the standard deviation

    random_smoothed = np.asarray([moving_average(row) for row in random_returns])
    random_mean = random_smoothed.mean(axis=0)
    random_std = random_smoothed.std(axis=0)

    plt.plot(episodes,random_mean,color="black",linestyle="--",label="uniform random",) #plotting the random agent learning curve
    #plotting the standard deviation for random agent
    plt.fill_between(episodes,random_mean - random_std,random_mean + random_std,color="black",alpha=0.10,) 
    #plotting a dotted line showing the target return threshold
    plt.axhline(TARGET_RETURN,linestyle=":",color="red",label=f"target={TARGET_RETURN}",) 
    plt.xlabel("Episode")
    plt.ylabel(f"Return ({WINDOW}-episode moving average)")
    plt.title("SARSA(λ) Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sarsa_lambda_learning_curves.png", dpi=200)
    plt.close()

#the function returns two values for each lamda, seed pair showing the number of episode it took the agent to first reach the target threshold 
#and the last window mean average return.
def summary_stats(matrix):
    matrix = np.asarray(matrix)
    smoothed = np.asarray([moving_average(row) for row in matrix])
    mean_curve = smoothed.mean(axis=0)
    hits = np.where(mean_curve >= TARGET_RETURN)[0]

    if len(hits) == 0:
        target_text = "not reached"
    else:
        target_text = str(int(hits[0] + WINDOW))

    mean_final = np.mean(matrix[:, -WINDOW:])

    return target_text, mean_final

#the function just prints the summary table calculated in the summary_stats function
def print_summary_table(all_returns, random_returns):
    print("\n----------------------------------")
    print("SUMMARY")
    print("----------------------------------")
    print(f"Target return = {TARGET_RETURN}")
    print(f"Moving-average window = {WINDOW}")
    print()

    print(
        f"{'setting':<12}"
        f"{'episodes to target':<22}"
        f"{'mean final return':<20}"
    )

    print("-" * 54)

    for lam in LAMBDAS:
        matrix = np.asarray(all_returns[lam])
        target_text, mean_final = summary_stats(matrix)

        print(
            f"{str(lam):<12}"
            f"{target_text:<22}"
            f"{mean_final:<20.3f}"
        )

#Finally the agent prints a sample episode with greedy policy using the trained agent
#we have manually selected λ = 0.3 (selected based on fastest target return reached) and seed = 0 (for reproducibility)
def print_greedy_episode():
    print("\n----------------------------------")
    print("TRAINED GREEDY SAMPLE EPISODE")
    print("----------------------------------")

    env = gym.make(ENV_ID, render_mode="ansi")

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

    state, _ = env.reset(seed=0)
    total_return = 0.0

    print(env.render())

    #exploration is turned off for the sample greedy episode because the agent has already gone through SARSA(lambda) training
    for step in range(1, 301):
        action = agent.eps_greedy(state,exploration=False,)
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

#runs all the functions in sequence
def main():
    all_returns = run_lambda_sweep()
    random_returns = run_random_baseline()

    plot_learning_curves(
        all_returns,
        random_returns,
    )

    print_summary_table(
        all_returns,
        random_returns,
    )

    print_greedy_episode()

    print("\nSaved plot: sarsa_lambda_learning_curves.png")


if __name__ == "__main__":
    main()