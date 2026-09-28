"""Task 2: SARSA(lambda) with eligibility traces.

Do not change the class name or the constructor signature -- the grading harness
constructs this class directly, and it will hand you an environment you have
never seen. Read the sizes off the spaces and never assume anything about what a
state number means. Do not import myenv from this file.
"""

from typing import Any

import numpy as np
import gymnasium as gym

ACCUMULATING = "accumulating"
REPLACING = "replacing"


def argmax_action(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return the index of the largest value, breaking ties uniformly at random.

    Ties are not an edge case here. The table starts uniform, so on the first
    visit to a state every action is tied, and a plain np.argmax would commit
    every state in the table to action 0.

    Args:
        values: the q-values of one state, shape (n_actions,)
        rng: the agent's random generator

    Returns:
        int: an action
    """
    # find the largest q value
    max_value = np.max(values)

    # find all actions with the largest q value
    best_actions = np.flatnonzero(values == max_value)

    # randomly choose an action from the best actions
    return int(rng.choice(best_actions))


class SarsaLambdaAgent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        alpha: float = 0.05,
        eps: float = 0.1,
        lam: float = 0.9,
        trace: str = ACCUMULATING,
        total_epi: int = 5_000,
        init_val: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            env: any tabular gym environment. Both spaces are Discrete.
            gamma: discount factor.
            alpha: learning rate.
            eps: exploration rate for a plain (non-decaying) epsilon-greedy.
            lam: the lambda of SARSA(lambda), in [0, 1]. At 0 this must reduce
                to ordinary one-step SARSA.
            trace: "replacing" or "accumulating".
            total_epi: number of training episodes.
            init_val: value every q(s,a) starts at. Setting this at or slightly
                above the best achievable return makes every untried action look
                good, which drives systematic exploration -- on a sparse-reward
                environment that is often what makes learning possible at all.
            seed: seed for the agent's own randomness, for reproducible runs.
        """
        if trace not in (ACCUMULATING, REPLACING):
            raise ValueError(f"unknown trace type: {trace}")

        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.lam = lam
        self.trace = trace
        self.total_epi = total_epi
        self.init_val = init_val
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.q = self.init_qtable(init_val)

    def init_qtable(self, init_val: float = 0.0) -> np.ndarray:
        """Build the q table, shape (n_states, n_actions), filled with init_val."""
        # create the q table using number of states and actions
        return np.full(
            (self.n_states, self.n_actions),
            init_val,
            dtype=float,
        )

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """Epsilon-greedy action selection over the current q table.

        Args:
            state: the current state
            exploration: explore with probability eps if True; act greedily if
                False. The greedy path is what best_run uses.

        Returns:
            int: an action
        """
        # choose a random action with probability eps when exploring
        if exploration and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))

        # otherwise, choose the action with the highest Q-value
        return argmax_action(self.q[state], self.rng)

    def learn(self) -> list[float]:
        """Run SARSA(lambda) for self.total_epi episodes, updating self.q.

        Returns:
            list[float]: the undiscounted return of each training episode, in
            order. myrunner.py plots these.
        """
        # store the total return from each episode
        returns = []

        # train for the specified number of episodes
        for episode in range(self.total_epi):

            # create a reproducible seed for each episode
            episode_seed = None if self.seed is None else self.seed + episode
            state, _ = self.env.reset(seed=episode_seed)

            # reset eligibility traces to 0 at the start of each episode.
            eligibility = np.zeros_like(self.q)

            # choose the first action using epsilon-greedy
            action = self.eps_greedy(state)

            # keep track of the total return for the episode
            total_return = 0.0

            while True:
                # take selected action and observe the next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # add the reward to the total return
                total_return += reward

                # if in terminal state then there is no future Q-value
                if terminated:
                    # calculate the TD error
                    delta = reward - self.q[state, action]
                    next_action = None

                else:
                    # choose the next action using epsilon-greedy
                    next_action = self.eps_greedy(next_state)

                    # calculate sarsa TD error
                    delta = (
                        reward
                        + self.gamma * self.q[next_state, next_action]
                        - self.q[state, action]
                    )

                # update current eligibility trace for current state-action pair based on the trace type
                if self.trace == ACCUMULATING:
                    eligibility[state, action] += 1.0
                else:  
                    eligibility[state, action] = 1.0

                # update all Q-values using the eligibility traces
                self.q += self.alpha * delta * eligibility

                # decay all eligibility traces
                eligibility *= self.gamma * self.lam

                # stop when the episode terminates or is truncated
                if terminated or truncated:
                    break

                # move to the next state-action pair
                state = next_state
                action = next_action

            # save the total return for the episode
            returns.append(total_return)

        return returns

    def best_run(self, max_steps: int = 300) -> tuple[list[tuple[int, int, float]], bool]:
        """Generate one greedy episode under the learned q table, for the report.

        Args:
            max_steps: give up after this many steps.

        Returns:
            tuple[
                list[tuple[int,int,float]]: the episode, as [(s, a, r), ...]
                bool: True if it reached a terminal state, False if it ran out
            ]
        """
        # store each state, action, and reward from the episode
        episode = []
        # reset the environment to get starting state
        state, _ = self.env.reset(seed=self.seed)

        # run the learned policy up to the maximum number of steps
        for _ in range(max_steps):
            # choose the best action without exploration
            action = self.eps_greedy(state, exploration=False)

            # take the greedy action
            next_state, reward, terminated, truncated, _ = (
                self.env.step(action)
            )

            # store the current state, action, and reward
            episode.append((state, action, reward))

            # check if the episode reached a terminal state
            if terminated:
                return episode, True

            if truncated:
                return episode, False

            # move to the next state
            state = next_state

        return episode, False
        

    def calc_return(self, episode: list[tuple[Any, Any, float]], discounted: bool = False) -> float:
        """Return of an episode given as [(s, a, r), ...]."""
        # start the total episode return at 0
        total = 0.0

        # go through each reward in the episode
        for t, (_, _, reward) in enumerate(episode):
            # apply discounting if specified
            if discounted:
                total += (self.gamma ** t) * reward
            else:
                total += reward

        return total


class RandomAgent(SarsaLambdaAgent):
    """The baseline your agent has to beat. Already written; do not change it."""

    def learn(self) -> list[float]:
        returns = []
        for _ in range(self.total_epi):
            self.env.reset()
            total = 0.0
            while True:
                action = int(self.rng.integers(self.n_actions))
                _, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        return returns
