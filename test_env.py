from gymnasium.utils.env_checker import check_env
from myenv import MyEnv
import gymnasium as gym
import myenv

# check that the environment follows Gymnasium rules
env = MyEnv()
check_env(env)

print("Environment passed the Gymnasium check!")


# check registered environment and renderer
env = gym.make("cs272/GreenHouse-v0", render_mode="ansi")

obs, info = env.reset(seed=42)

print("\nStarting observation:", obs)
print("Info:", info)
print()
print(env.render())