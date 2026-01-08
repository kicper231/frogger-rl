import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from DQN.DQN_nets import DuelingDQN
import os
from datetime import datetime 
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
ACTION_NUM = 5 
WEIGHTS_PATH = "dqn\model3.weights.h5" 
FRAME_SKIP = 8 
gym.register_envs(ale_py)
model = DuelingDQN(ACTION_NUM) 
model(tf.zeros((1, 84, 84, 4), dtype=tf.float32)) 
model.load_weights(WEIGHTS_PATH)

def select_action(state): 
    state = np.transpose(state, (1, 2, 0)) 
    state = tf.convert_to_tensor(state, dtype=tf.float32) / 255.0 
    state = tf.expand_dims(state, axis=0) 
    q_values = model(state, training=False)[0] 
    return int(tf.argmax(q_values).numpy())
RUNS = 50
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

for run_idx in range(1, RUNS + 1):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    env = gym.make(
        "ALE/Frogger-v5",
        obs_type="grayscale",
        frameskip=1,
        render_mode="rgb_array",
    )

    env = AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, frame_skip=FRAME_SKIP
    )
    env = FrameStackObservation(env, stack_size=4)

    env = RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=f"frogger_run_{run_idx}",
        episode_trigger=lambda ep: True,
    )

    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    env.close()

    # ---- rename ostatniego wideo ----
    videos = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".mp4")],
        key=lambda f: os.path.getmtime(os.path.join(video_dir, f)),
    )

    if videos:
        last_video = os.path.join(video_dir, videos[-1])

        if total_reward > 20:
            new_name = (
                f"frogger_run_{run_idx}_"
                f"score_{int(total_reward)}_"
                f"time_{timestamp}.mp4"
            )
            new_path = os.path.join(video_dir, new_name)
            os.rename(last_video, new_path)
            print(f"📁 ZAPISANY: {new_name}")
        else:
            os.remove(last_video)
            print(f"🗑️ Wideo usunięte (za mało punktów) {total_reward}")

print("✅ Wszystkie 5 nagrań gotowe")
