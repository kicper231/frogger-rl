import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from DQN.DQN_nets import DuelingDQN  

ACTION_NUM = 5
WEIGHTS_PATH = "dqn/model2.weights.h5"
EPISODES = 5
FRAME_SKIP = 8  

gym.register_envs(ale_py)

env = gym.make(
    "ALE/Frogger-v5",
    obs_type="grayscale",
    frameskip=1,
    render_mode="human", 
)

env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=FRAME_SKIP)
env = FrameStackObservation(env, stack_size=4)

model = DuelingDQN(ACTION_NUM)

dummy = tf.zeros((1, 84, 84, 4), dtype=tf.float32)
model(dummy)

print(f"Ładowanie wag z: {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH)

def select_action(state):

    state = np.transpose(state, (1, 2, 0)) 
    state = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0
    state = tf.expand_dims(state, axis=0)
    q_values = model(state, training=False)[0]
    return int(tf.argmax(q_values).numpy())

for ep in range(EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = next_state
        steps += 1

    print(
        f"🎮 Episode {ep+1}/{EPISODES} | Reward = {total_reward:.2f} | Steps = {steps}"
    )

env.close()
