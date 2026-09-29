import numpy as np

N_STATES = 30 #observation space: 6 growth stages * 5 moisture stages
N_ACTIONS = 3 #action space: 0 water, 1 wait, 2 harvest
START_STATE = 2 #start: 0 growth, 2 moisture
GAMMA = 1.0 #discount factor

#the dynamic programming agent has complete knowledge of the environment and transition probabilities so that it can calculate the optimal 
#expected return directly without learning using sampled episodes.
def decode(state):
    return divmod(state, 5) #decodes 1 integer state into growth and moisture

def encode(growth, moisture):
    return growth * 5 + moisture #encodes the growth and moisture into 1 integer

def harvest_reward(growth): #harvest action at any growth stage terminates the episode
    if growth == 5:
        return 100.0 #optimal harvest
    if growth == 4:
        return 5.0 #mature harvest
    return -10.0 #premature harvest

def water_wait_outcomes(growth, moisture, action): 
    if action == 0:
        moisture_outcomes = [(0.80, 1), (0.20, 2)] #transition probabilities and +1 and +2 outcome to the moisture level at water action
    elif action == 1:
        moisture_outcomes = [(0.80, -1), (0.20, 0)] #transition probabilities and -1 or no chnge to the moisture level at wait action
    else:
        raise ValueError("action must be 0 or 1")

    outcomes = []

    for moisture_probability, change in moisture_outcomes:
        new_moisture = np.clip(moisture + change, 0, 4)
        reward = -5.0 if new_moisture in (0, 4) else -1.0 #immidiate reward of -5 in case of moisture level 0 and 4, else -1.

        #growth probability is stochastic: same action might not produce the same results are the growth chance is 50%
        if 1 <= new_moisture <= 3 and growth < 5:
            outcomes.append((moisture_probability * 0.50,encode(growth + 1, new_moisture),reward))
            outcomes.append((moisture_probability * 0.50,encode(growth, new_moisture),reward))
        else:
            outcomes.append((moisture_probability,encode(growth, new_moisture),reward))

    return outcomes

#calculates the expected value of taking an action in the given state. 
def action_value(value_function, state, action):
    growth, moisture = decode(state)
    if action == 2:
        return harvest_reward(growth)
    outcomes = water_wait_outcomes(growth, moisture, action)

    return sum(probability * (reward + GAMMA * value_function[next_state]) for probability, next_state, reward in outcomes)

#this function runs the dp agent to obtain optimal state value and best action policy for each state. 
#theta denotes the convergence threshold and max_iteration is kept so that the agent doesn't run forever.
def value_iteration(theta=1e-8, max_iterations=5000):
    values = np.zeros(N_STATES)
    start_value_history = []

    for iteration in range(1, max_iterations + 1):
        new_values = np.zeros(N_STATES)
        #examines every state and calculates expected value for each possible action in that state.
        for state in range(N_STATES):
            action_values = [action_value(values, state, action) for action in range(N_ACTIONS)]
            new_values[state] = max(action_values) 

        difference = np.max(np.abs(new_values - values))
        values = new_values
        start_value_history.append(values[START_STATE])

        if difference < theta: #if the the difference becomes lower than the theta value we stop the loop.
            break

    q_values = np.zeros((N_STATES, N_ACTIONS))

    for state in range(N_STATES):
        for action in range(N_ACTIONS):
            q_values[state, action] = action_value(values,state,action,)
    policy = np.argmax(q_values, axis=1) #selects the best action for that state
    return (values,q_values,policy,iteration,start_value_history)


if __name__ == "__main__":
    values, q_values, policy, iterations, history = value_iteration()

    action_names = {
        0: "Water",
        1: "Wait",
        2: "Harvest",
    }

    print(f"Value iteration completed in {iterations} iterations")
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
