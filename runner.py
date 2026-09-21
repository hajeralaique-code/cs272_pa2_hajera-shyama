import gymnasium as gym
import myenv
from myagent import SarsaLambdaAgent

env = gym.make("cs272/GreenHouse-v0")

agent = SarsaLambdaAgent(env,lam=0.9,total_epi=5000,seed=42,)

returns = agent.learn()
episode, reached_terminal = agent.best_run()