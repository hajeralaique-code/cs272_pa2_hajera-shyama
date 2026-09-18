"""A stochastic greenhouse environment.

The agent manages a growing plant by choosing whether to water, wait, or
harvest. The goal is to keep soil moisture in a healthy range long enough for
the plant to reach full maturity, then harvest it for the largest reward.

Extremely dry or wet soil stresses the plant and receives a penalty.
Random variation in irrigation and plant growth makes the environment stochastic."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class MyEnv(gym.Env):
    """Manage soil moisture and grow a plant to full maturity before harvesting."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # Growth stages: 0=seed, 1-4=growing, 5=fully ripe.
        self.num_growth_stages = 6

        # Moisture levels: 0=bone dry, 1-3=healthy range, 4=waterlogged.
        self.num_moisture_levels = 5

        # State is encoded as:
        # state = growth_stage * 5 + moisture
        # This gives 6 * 5 = 30 possible states.
        self.observation_space = spaces.Discrete(30)

        # Actions:
        # 0 = Water
        # 1 = Wait
        # 2 = Harvest
        self.action_space = spaces.Discrete(3)

        # These hold the current state of the greenhouse.
        self.growth_stage = 0
        self.moisture = 2

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

    # Two helper functions
    def _get_obs(self):
        return int(self.growth_stage * 5 + self.moisture)

    def _get_info(self):
        return {
            "growth_stage": self.growth_stage,
            "moisture": self.moisture,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)

        self.growth_stage = 0
        self.moisture = 2

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.

        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        terminated = False
        reward = -0.1

        # Action 2: Harvest
        if action == 2:
            terminated = True

            if self.growth_stage == 5:
                reward = 100.0
            elif self.growth_stage == 4:
                reward = 10.0
            else:
                reward = -1.0

            return self._get_obs(), reward, terminated, False, self._get_info()

        # Actions 0 and 1: Water or Wait
        transition_roll = self.np_random.random()

        if action == 0:  # Water
            if transition_roll < 0.80:
                moisture_change = 1
            else:
                moisture_change = 2

        elif action == 1:  # Wait
            if transition_roll < 0.80:
                moisture_change = -1
            else:
                moisture_change = 0

        new_moisture = self.moisture + moisture_change

        # Keep moisture inside the valid range 0-4.
        new_moisture = max(0, min(4, new_moisture))
        self.moisture = new_moisture

        # Reward depends on the resulting moisture.
        if self.moisture == 0 or self.moisture == 4:
            reward = -1.0
        else:
            reward = -0.1

        # Growth is only possible when moisture is healthy.
        if 1 <= self.moisture <= 3 and self.growth_stage < 5:
            if self.np_random.random() < 0.80:
                self.growth_stage += 1

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None
        # TODO: draw it. You need this for the sample episode in your report.
        raise NotImplementedError

    def close(self):
        pass


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/MyEnv-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
