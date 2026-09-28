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

# symbols used to visually display growth and moisture bars in render()
FULL, EMPTY = "█", "░"  

# names for each growth stage and moisture level
STAGE_NAMES = ["Seed", "Sprout", "Seedling", "Growing", "Budding", "Ripe"]
MOISTURE_NAMES = [
    "Bone dry (no growth)", "Healthy", "Healthy", "Healthy",
    "Waterlogged (no growth)",
]

# ASCII drawings used to visually display the plant at each growth stage 
PLANT_ART = [
    # stage 0: seed
    ["         ", "         ", "         ", "    .    "],
    # stage 1: sprout
    ["         ", "         ", "   \\ /   ", "    |    "],
    # stage 2: seedling
    ["         ", "  \\ | /  ", "   \\|/   ", "    |    "],
    # stage 3: growing
    ["    |    ", "  \\ | /  ", "   \\|/   ", "    |    "],
    # stage 4: budding
    ["    o    ", "  \\ | /  ", "   \\|/   ", "    |    "],
    # stage 5: ripe
    ["   (*)   ", "  \\ | /  ", "   \\|/   ", "    |    "],
]
# ASCII drawing used to visually display the pot
POT_ART = ["[=======]", " \\_____/ "]


class MyEnv(gym.Env):
    """Manage soil moisture and grow a plant to full maturity before harvesting."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # growth stages: 0=seed, 1-4=growing, 5=fully ripe.
        self.num_growth_stages = 6

        # moisture levels: 0=bone dry, 1-3=healthy range, 4=waterlogged.
        self.num_moisture_levels = 5

        # state is encoded as:
        # state = growth_stage * 5 + moisture
        # therefore we have 6 * 5 = 30 possible states.
        self.observation_space = spaces.Discrete(30)

        # our actions are:
        # 0 = water, 1 = wait, 2 = harvest
        self.action_space = spaces.Discrete(3)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

    # two helper functions
    # convert growth stage and moisture to observation
    def _get_obs(self):
        return int(self.growth_stage * 5 + self.moisture)

    # return the individial state values as info
    def _get_info(self):
        return {
            "growth_stage": self.growth_stage,
            "moisture": self.moisture,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        # this line seeds self.np_random
        # for reproducibile stochastic behavior
        super().reset(seed=seed)

        # start a new episode at growth stage 0 and moisture level 2
        self.growth_stage = 0
        self.moisture = 2

        return self._get_obs(), self._get_info()

    def step(self, action: int):        
        # reject actions outside the valid range
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        # episodes continue by default until the agent harvests
        terminated = False
        reward = -0.1

        # action 2: harvest
        if action == 2:
            # terminate always at harvest regardless of growth stage or moisture level
            terminated = True

            # determine reward based on growth stage
            if self.growth_stage == 5:
                reward = 100.0
            elif self.growth_stage == 4:
                reward = 5.0
            else:
                reward = -10.0

            return self._get_obs(), reward, terminated, False, self._get_info()

        # actions 0 and 1: water or wait
        # randomness for transitions
        transition_roll = self.np_random.random()

        # water
        if action == 0:  
            if transition_roll < 0.80:
                moisture_change = 1
            else:
                moisture_change = 2    
        # wait
        elif action == 1:  
            if transition_roll < 0.80:
                moisture_change = -1
            else:
                moisture_change = 0

        # apply the randomly determined moisture change
        new_moisture = self.moisture + moisture_change

        # make sure moisture level is within bounds
        new_moisture = max(0, min(4, new_moisture))
        self.moisture = new_moisture

        # give larger penalty for extreme moisture levels
        if self.moisture == 0 or self.moisture == 4:
            reward = -5.0         
        else:
            reward = -1.0          

        # growth is only possible when moisture is in healthy range
        if 1 <= self.moisture <= 3 and self.growth_stage < 5:
            # when in healthy range plant has 50% chance of growing
            if self.np_random.random() < 0.50:
                self.growth_stage += 1

        # added the gymnasium step return values
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None

        # determine the maximum values for constructing the display bars
        max_stage = self.num_growth_stages - 1       
        max_moisture = self.num_moisture_levels - 1  

        # build the bars: filled slots is the current value, empty slots means theres room left.
        growth_bar = FULL * self.growth_stage + EMPTY * (max_stage - self.growth_stage)
        moisture_bar = FULL * self.moisture + EMPTY * (max_moisture - self.moisture)
        # keep columns aligned
        moisture_bar = moisture_bar.ljust(max_stage)  

        # combine the plant and plot drawing for left side
        picture = PLANT_ART[self.growth_stage] + POT_ART

        # labels showing growth and moisture levels for the state on the right side
        notes = [
            f"Growth    {growth_bar}  {self.growth_stage}/{max_stage}  "
            f"{STAGE_NAMES[self.growth_stage]}",
            "",
            f"Moisture  {moisture_bar}  {self.moisture}/{max_moisture}  "
            f"{MOISTURE_NAMES[self.moisture]}",
            "",
            f"State {self._get_obs()}  "
            f"(stage {self.growth_stage}, moisture {self.moisture})",
            "",
        ]

        # place plant drawing ans state info side by side
        rows = [f"{art}   {note}".rstrip() for art, note in zip(picture, notes)]
        return "\n".join(["=== Greenhouse ===", *rows])
    

    def close(self):
        pass



# added a custom greenhouse environment name
register(
    id="cs272/GreenHouse-v0",
    entry_point="myenv:MyEnv",
    # kept the original max_episode_steps as 300
    max_episode_steps=300,
)
