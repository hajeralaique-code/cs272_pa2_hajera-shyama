"""Exact dynamic programming solution for the greenhouse MDP.

Unlike SarsaLambdaAgent, which only ever sees samples from env.step(), this
script is handed the environment's full transition model directly (the
water/wait probabilities, the growth probability, the reward rules) and uses
it to compute the optimal value function exactly, by value iteration. There
is nothing to "learn" here -- it is the ceiling SARSA(lambda) is trying to
reach through trial and error.

This mirrors myenv.py's step() logic. If you change the environment's
rewards or probabilities, update the matching lines here too.
"""

import numpy as np

N_STATES = 30    # growth_stage (0-5) * 5 + moisture (0-4)
N_ACTIONS = 3    # 0 water, 1 wait, 2 harvest
START_STATE = 2  # growth_stage=0, moisture=2

# Undiscounted, to match the raw episode returns SARSA's learning curve plots.
# Well-posed here because harvest is always available and gives a finite
# reward, so the optimal policy is guaranteed to terminate.
GAMMA = 1.0


def decode(state: int) -> tuple[int, int]:
    return divmod(state, 5)


def encode(growth: int, moisture: int) -> int:
    return growth * 5 + moisture


def harvest_reward(growth: int) -> float:
    if growth == 5:
        return 100.0
    if growth == 4:
        return 5.0
    return -10.0


def water_wait_outcomes(growth: int, moisture: int, action: int):
    """All (probability, next_state, reward) outcomes for action 0 or 1."""
    if action == 0:  # water
        moisture_branches = [(0.8, 1), (0.2, 2)]
    else:  # wait
        moisture_branches = [(0.8, -1), (0.2, 0)]

    outcomes = []
    for p_m, delta in moisture_branches:
        new_moisture = max(0, min(4, moisture + delta))
        reward = -5.0 if new_moisture in (0, 4) else -1.0

        if 1 <= new_moisture <= 3 and growth < 5:
            for p_g, grows in [(0.5, True), (0.5, False)]:
                new_growth = growth + 1 if grows else growth
                outcomes.append((p_m * p_g, encode(new_growth, new_moisture), reward))
        else:
            outcomes.append((p_m, encode(growth, new_moisture), reward))

    return outcomes


def value_iteration(theta: float = 1e-8, max_iters: int = 10_000):
    """Return (V, Q, policy, n_iters, start_value_history).

    start_value_history is V[START_STATE] after every sweep -- this is DP's
    own "learning curve": how its estimate of the start state's value
    settles over iterations, for plotting next to SARSA's episode curve.
    """
    V = np.zeros(N_STATES)
    start_value_history = []

    for iteration in range(1, max_iters + 1):
        new_V = np.copy(V)
        delta = 0.0

        for state in range(N_STATES):
            growth, moisture = decode(state)
            q_water = sum(p * (r + GAMMA * V[s2]) for p, s2, r in water_wait_outcomes(growth, moisture, 0))
            q_wait = sum(p * (r + GAMMA * V[s2]) for p, s2, r in water_wait_outcomes(growth, moisture, 1))
            q_harvest = harvest_reward(growth)

            new_V[state] = max(q_water, q_wait, q_harvest)
            delta = max(delta, abs(new_V[state] - V[state]))

        V = new_V
        start_value_history.append(V[START_STATE])
        if delta < theta:
            break

    Q = np.zeros((N_STATES, N_ACTIONS))
    for state in range(N_STATES):
        growth, moisture = decode(state)
        Q[state, 0] = sum(p * (r + GAMMA * V[s2]) for p, s2, r in water_wait_outcomes(growth, moisture, 0))
        Q[state, 1] = sum(p * (r + GAMMA * V[s2]) for p, s2, r in water_wait_outcomes(growth, moisture, 1))
        Q[state, 2] = harvest_reward(growth)

    policy = np.argmax(Q, axis=1)
    return V, Q, policy, iteration, start_value_history


if __name__ == "__main__":
    V, Q, policy, n_iters, start_value_history = value_iteration()
    action_names = {0: "water", 1: "wait", 2: "harvest"}

    print(f"Value iteration converged in {n_iters} sweeps.\n")
    print("Optimal policy, (growth, moisture) -> action  [V]:")
    for state in range(N_STATES):
        growth, moisture = decode(state)
        print(f"  ({growth}, {moisture}) -> {action_names[policy[state]]:8s} V={V[state]:7.2f}")

    print(f"\nOptimal expected return from the start state: {V[START_STATE]:.3f}")