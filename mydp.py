import numpy as np


N_STATES = 30
N_ACTIONS = 3
START_STATE = 2
GAMMA = 1.0


def decode(state):
    return divmod(state, 5)


def encode(growth, moisture):
    return growth * 5 + moisture


def harvest_reward(growth):
    if growth == 5:
        return 100.0
    if growth == 4:
        return 5.0
    return -10.0


def water_wait_outcomes(growth, moisture, action):
    if action == 0:
        moisture_outcomes = [(0.80, 1), (0.20, 2)]
    elif action == 1:
        moisture_outcomes = [(0.80, -1), (0.20, 0)]
    else:
        raise ValueError("action must be 0 or 1")

    outcomes = []

    for moisture_probability, change in moisture_outcomes:
        new_moisture = np.clip(moisture + change, 0, 4)
        reward = -5.0 if new_moisture in (0, 4) else -1.0

        if 1 <= new_moisture <= 3 and growth < 5:
            outcomes.append((
                moisture_probability * 0.50,
                encode(growth + 1, new_moisture),
                reward,
            ))
            outcomes.append((
                moisture_probability * 0.50,
                encode(growth, new_moisture),
                reward,
            ))
        else:
            outcomes.append((
                moisture_probability,
                encode(growth, new_moisture),
                reward,
            ))

    return outcomes


def action_value(value_function, state, action):
    growth, moisture = decode(state)

    if action == 2:
        return harvest_reward(growth)

    outcomes = water_wait_outcomes(growth, moisture, action)

    return sum(
        probability * (
            reward + GAMMA * value_function[next_state]
        )
        for probability, next_state, reward in outcomes
    )


def value_iteration(theta=1e-8, max_iterations=10000):
    values = np.zeros(N_STATES)
    start_value_history = []

    for iteration in range(1, max_iterations + 1):
        new_values = np.zeros(N_STATES)

        for state in range(N_STATES):
            action_values = [
                action_value(values, state, action)
                for action in range(N_ACTIONS)
            ]

            new_values[state] = max(action_values)

        difference = np.max(np.abs(new_values - values))
        values = new_values
        start_value_history.append(values[START_STATE])

        if difference < theta:
            break

    q_values = np.zeros((N_STATES, N_ACTIONS))

    for state in range(N_STATES):
        for action in range(N_ACTIONS):
            q_values[state, action] = action_value(
                values,
                state,
                action,
            )

    policy = np.argmax(q_values, axis=1)

    return (
        values,
        q_values,
        policy,
        iteration,
        start_value_history,
    )


if __name__ == "__main__":
    values, q_values, policy, iterations, history = value_iteration()

    action_names = {
        0: "Water",
        1: "Wait",
        2: "Harvest",
    }

    print(f"Value iteration converged in {iterations} iterations")
    print(f"Optimal value from start state: {values[START_STATE]:.3f}\n")

    for state in range(N_STATES):
        growth, moisture = decode(state)

        print(
            f"State {state:2d}: "
            f"growth={growth}, "
            f"moisture={moisture}, "
            f"action={action_names[policy[state]]}, "
            f"value={values[state]:.3f}"
        )