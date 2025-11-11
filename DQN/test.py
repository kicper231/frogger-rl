import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np

from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from model_DQN import DuelingDQN  # Twój model

# ---------- ŁADOWANIE MODELU ----------
ACTION_NUM = 5
model = DuelingDQN(ACTION_NUM)

# Dummy forward pass
dummy = tf.zeros((1, 84, 84, 4), dtype=tf.float32)
model(dummy)

# Załaduj wagi
print("Loading weights...")
model = tf.keras.models.load_model("dqn_dueling_full_model.keras", compile=False)

print("Weights loaded ✅")


# ---------- FUNKCJA WYBORU AKCJI ----------
def select_action(state):
    """
    Stacked 4 frames: shape (4, 84, 84)
    Convert to (1, 84, 84, 4) and calculate argmax.
    """
    state = np.transpose(state, (1, 2, 0))  # CHW → HWC
    state = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0
    state = tf.expand_dims(state, axis=0)

    q_values = model(state, training=False)[0]
    action = tf.argmax(q_values).numpy()
    return int(action)


# ---------- ŚRODOWISKO ----------
gym.register_envs(ale_py)

env = gym.make("ALE/Frogger-v5", obs_type="grayscale", frameskip=1)
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
env = FrameStackObservation(env, stack_size=4)


# ---------- RUN TEST ----------
episodes_to_play = 10

for ep in range(episodes_to_play):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # 🔥 pokazuje obraz
        action = select_action(state)
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward

    print(f"Episode {ep+1}/{episodes_to_play} reward = {total_reward}")

env.close()
