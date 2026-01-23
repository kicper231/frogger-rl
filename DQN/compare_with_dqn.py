import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import gymnasium as gym
import ale_py
from stable_baselines3.common.callbacks import EvalCallback


gym.register_envs(ale_py)

env = gym.make("ALE/Frogger-v5")
env = AtariWrapper(env)

model = DQN(
    policy="CnnPolicy",
    env=env,
    verbose=1,
)

eval_env = gym.make("ALE/Frogger-v5")
eval_env = AtariWrapper(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best/",
    log_path="./eval_logs/",
    eval_freq=100_000,
    n_eval_episodes=5,
    deterministic=True,
)

model.learn(total_timesteps=10000000, callback=eval_callback)

model.save("frogger_dqn")
