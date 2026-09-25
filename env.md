# GreenHouse Environment

## Story

GreenHouse is a stochastic plant-growing environment. The agent manages a plant by choosing whether to water, wait, or harvest. The goal is to keep the soil moisture in a healthy range long enough for the plant to reach full maturity, then harvest it for the largest reward.

Very dry or waterlogged soil stresses the plant and gives a penalty. Irrigation and plant growth are stochastic, so the same action in the same state does not always produce the same result.

## Observation Space

The observation space is:

```python
spaces.Discrete(30)
```

There are 6 growth stages and 5 moisture levels, giving 30 possible states.

The state is encoded as:

```text
state = growth_stage * 5 + moisture
```

### Growth Stages

| Value | Stage |
|---|---|
| 0 | Seed |
| 1 | Sprout |
| 2 | Seedling |
| 3 | Growing |
| 4 | Budding |
| 5 | Ripe |

### Moisture Levels

| Value | Meaning |
|---|---|
| 0 | Bone dry |
| 1 | Healthy |
| 2 | Healthy |
| 3 | Healthy |
| 4 | Waterlogged |

For example, growth stage 2 with moisture level 3 is encoded as:

```text
2 * 5 + 3 = 13
```

## Action Space

The action space is:

```python
spaces.Discrete(3)
```

| Action | Meaning |
|---|---|
| 0 | Water |
| 1 | Wait |
| 2 | Harvest |

## Reward Structure

For Water and Wait:

| Resulting Moisture | Reward |
|---|---:|
| 0 or 4 | -5 |
| 1, 2, or 3 | -1 |

For Harvest:

| Growth Stage | Reward |
|---|---:|
| 5 (Ripe) | +100 |
| 4 (Budding) | +5 |
| 0 through 3 | -10 |

The largest reward is received by keeping the plant healthy until it reaches the ripe stage and then harvesting it.

## Starting State

Every episode begins at:

```text
growth_stage = 0
moisture = 2
```

Therefore, the starting observation is:

```text
state = 0 * 5 + 2 = 2
```

## Termination and Truncation

The episode terminates when the agent chooses the Harvest action.

The environment does not set truncation itself. Gymnasium's `TimeLimit` wrapper handles truncation after 300 steps.

```python
max_episode_steps=300
```

## Transition Noise

All randomness in the environment comes from `self.np_random`.

### Water

When the agent chooses Water:

- 80% probability that moisture increases by 1
- 20% probability that moisture increases by 2

### Wait

When the agent chooses Wait:

- 80% probability that moisture decreases by 1
- 20% probability that moisture stays the same

Moisture is always kept within the range 0 through 4.

### Plant Growth

The plant can only grow when moisture is in the healthy range from 1 through 3.

When moisture is healthy and the plant has not reached stage 5:

- 50% probability that the growth stage increases by 1
- 50% probability that the growth stage stays the same

The maximum growth stage is 5.

## Constructor Arguments

The environment constructor is:

```python
MyEnv(render_mode=None)
```

The supported render mode is `"ansi"`. When `render_mode="ansi"`, `render()` returns a human-readable representation of the plant, moisture level, growth stage, and encoded state.

## Registered Environment ID

The environment is registered as:

```text
cs272/GreenHouse-v0
```

It can be created with:

```python
import gymnasium as gym
import myenv

env = gym.make("cs272/GreenHouse-v0")
```

For ANSI rendering:

```python
env = gym.make(
    "cs272/GreenHouse-v0",
    render_mode="ansi",
)
```