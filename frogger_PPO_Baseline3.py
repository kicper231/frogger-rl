import gymnasium as gym
import ale_py  # <-- required to register ALE envs
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import FrameStackObservation

env = gym.make("ALE/Frogger-v5", render_mode=None)  # stickiness in v5
# SB3’s AtariWrapper already does: grayscale, resize(84x84), frame-skip=4, clip rewards etc.
env = AtariWrapper(env)
# ensure 4-frame stack (AtariWrapper does frame-stack, but if you prefer Gymnasium’s:)
# env = FrameStackObservation(env, stack_size=4)

model = DQN(
    "CnnPolicy",
    env,
)
model.learn(total_timesteps=1_000_000)
