# CS272 PA2 - Greenhouse Environment and SARSA(lambda)

PA2 submission for Hajera Laique and Kumari Shyama.

## Project Overview

This project implements a stochastic greenhouse environment using Gymnasium and a SARSA(lambda) reinforcement learning agent with eligibility traces. The agent learns how to manage a growing plant by choosing when to water, wait, or harvest.

The goal is to maintain healthy soil moisture while allowing the plant to grow through its stages and eventually harvest it when it is fully ripe.

## Environment Overview

The greenhouse has 6 growth stages and 5 moisture levels.

The observation space contains 30 discrete states. Each state is encoded as:

`state = growth_stage * 5 + moisture`

The action space contains 3 actions:

- `0` = water
- `1` = wait
- `2` = harvest

The plant begins at growth stage 0 with moisture level 2.

Water and wait actions contain stochastic transitions. Water increases moisture by 1 with 80% probability and by 2 with 20% probability. Wait decreases moisture by 1 with 80% probability and leaves it unchanged with 20% probability. When moisture is in the healthy range of 1-3, the plant has a 50% chance of advancing one growth stage.

The reward structure encourages the agent to maintain healthy moisture and harvest at full maturity. A ripe harvest receives the largest reward, while unhealthy moisture and premature harvesting receive penalties.

An episode terminates when the agent chooses to harvest. Episodes are truncated by Gymnasium's `TimeLimit` after 300 steps.

The environment constructor accepts the optional `render_mode` argument and supports `render_mode="ansi"`.

Registered environment:

`cs272/GreenHouse-v0`

For the complete environment documentation, including the full reward structure, observation encoding, transition probabilities, starting state, termination conditions, and rendering information, see **[env.md](env.md)**.

## Main Files

- `myenv.py` - implements and registers the stochastic greenhouse Gymnasium environment.
- `myagent.py` - implements the SARSA(lambda) agent with accumulating and replacing eligibility traces, epsilon-greedy action selection, and the random-agent baseline.
- `myrunner.py` - runs the lambda sweep, trains across multiple seeds, generates the learning curves and summary results, and prints a sample greedy episode.
- `env.md` - contains the detailed documentation for the greenhouse environment.
- `requirements.txt` - lists the Python dependencies required to run the project.

## SARSA(lambda) Experiment

The SARSA(lambda) agent is evaluated using:

`lambda = {0.0, 0.3, 0.6, 0.9, 1.0}`

Each lambda value is trained using 5 different random seeds. The results are used to compare learning speed and final performance across lambda values.

The learning curves use a 100-episode moving average and show the mean and spread across seeds.

## Extra Implementations

The files below include additional experiments and analysis beyond the main implementation.

- `myenv_sparse.py` - sparse-reward version of the greenhouse environment where reward is delayed until a successful ripe harvest.
- `myrunner_sparse.py` - runs the SARSA(lambda) experiments on the sparse-reward greenhouse and generates additional learning and episode plots.
- `mydp.py` - contains the additional dynamic programming agent implementation.
- `myrunner_dp.py` - runs the dynamic programming agent performance evaluation in comparison with SARSA(lambda) agent performance.
- `test_env.py` - contains testing and correctness checks for our environment.

These files are supplemental implementations and are separate from the main `myenv.py`, `myagent.py`, and `myrunner.py` submission.

## Running the Project

```bash
pip install -r requirements.txt
python myrunner.py
