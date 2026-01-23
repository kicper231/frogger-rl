import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from collections import deque
from DQN_nets import DQN, DuelingDQN
from DQN_config import Config
from DQN_bufor import ReplayBuffer
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def make_env():
    gym.register_envs(ale_py)

    env = gym.make("ALE/Frogger-v5", obs_type="grayscale", frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=8)
    env = FrameStackObservation(env, stack_size=4)

    return env

def build_models(action_num: int):
    tf.keras.backend.clear_session()

    if (Config.DUELING):
        main_nn = DuelingDQN(action_num)
        target_nn = DuelingDQN(action_num)
    else:
        main_nn = DQN(action_num)
        target_nn = DQN(action_num)

    # keras need pass forward at least once
    zeros = tf.zeros((1, 84, 84, 4), dtype=tf.float32)
    main_nn(zeros)
    target_nn(zeros)

    target_nn.set_weights(main_nn.get_weights())
    return main_nn, target_nn


@tf.function
def train_step(
    main_nn, target_nn, optimizer, loss_fn, states, actions, rewards, next_states, dones
):
    # double dqn
    next_q_main = main_nn(next_states, training=False)
    next_actions = tf.argmax(next_q_main, axis=1)

    next_q_target = target_nn(next_states, training=False)
    next_action_masks = tf.one_hot(next_actions, Config.ACTION_NUM)
    max_next_qs = tf.reduce_sum(next_action_masks * next_q_target, axis=1)

    # bellman equation
    # TODO (n-step dqn)
    target = (
        rewards + (1.0 - tf.cast(dones, tf.float32)) * Config.DISCOUNT * max_next_qs
    )

    with tf.GradientTape() as tape:
        qs = main_nn(states, training=True)
        action_masks = tf.one_hot(actions, Config.ACTION_NUM)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=1)
        loss = loss_fn(target, masked_qs)

    grads = tape.gradient(loss, main_nn.trainable_variables)
    grads = [tf.clip_by_norm(g, 10.0) for g in grads]
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

    return loss


# not used - nie zauważono poprawy
def soft_update(target, online, tau=0.003):
    target_weights = target.get_weights()
    online_weights = online.get_weights()

    new_weights = []
    for tw, ow in zip(target_weights, online_weights):
        new_weights.append(tau * ow + (1 - tau) * tw)

    target.set_weights(new_weights)

def preprocess_state_for_nn(state):
    # state: (4, 84, 84) -> (1, 84, 84, 4) float32 [0,1]
    state = tf.transpose(state, perm=[1, 2, 0])
    state = tf.convert_to_tensor(state, dtype=tf.uint8)
    state = tf.expand_dims(state, axis=0)
    state = tf.cast(state, tf.float32) / 255.0
    return state


def select_greedy(env, main_nn, state, epsilon=Config.EPSILON):
    if  tf.random.uniform(()) < epsilon:
        return env.action_space.sample()
    else:
        state = preprocess_state_for_nn(state)
        q = main_nn(state, training=False)
        return int(tf.argmax(q[0]).numpy())

def init_results_file():
    with open(f"{RESULTS_DIR}/results_{Config.MODEL_NAME}.txt", "a") as f:
        f.write("\n=== CONFIG ===\n")
        for k, v in Config.__dict__.items():
            if not k.startswith("_"):
                f.write(f"{k}={v}\n")
        f.write("=== LOG ===\n")
        f.write("iter, episode, ep_real_reward, reward, loss, epsilon \n")

def append_results(iteration, episode, ep_real_reward, reward, loss, epsilon):
    with open(f"{RESULTS_DIR}/results_{Config.MODEL_NAME}.txt", "a") as f:
        f.write(f"{iteration}, {episode}, {ep_real_reward}, {reward}, {loss}, {epsilon} \n")

def run_training():
    env = make_env()

    main_nn, target_nn = build_models(action_num=5)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=Config.LEARNING_RATE, epsilon=1e-5
    )
    loss_fn = tf.keras.losses.Huber(delta=1.0)
    iteration = 1
    buffer = ReplayBuffer(Config.BUFFERSIZE)
    epsilon = Config.EPSILON

    avg_window = deque(maxlen=100)
    avg_real = deque(maxlen=100)
    best = 0

    init_results_file()

    for episode in range(Config.NUM_EPISODES + 1):
        state, info = env.reset()
        done = False

        loss = 0
        ep_reward = 0
        ep_real_reward = 0

        ep_iter = 0
        lifes = 4  # licznik żyć

        while not done:
            action = select_greedy(env, main_nn, state, epsilon)

            # warmup
            if ep_iter < 30:
                obs, reward, terminated, truncated, info = env.step(0)
                state = obs
                ep_iter += 1
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_real_reward += reward
            
            if Config.REWARD_NORMALISATION:
                reward = np.sign(reward)

            # kara za utrate życia
            if lifes != info["lives"]:
                lifes -= 1
                reward -= Config.DEATHPENALTY

            # Revard Shaping
            if action == 4:
                reward -= Config.BACKWARDS_PENALTY
            if action == 0:
                reward -= Config.MOVE_IN_PLACE_PENALTY

            buffer.add(state, obs, action, reward, done)
            ep_reward += reward

            iteration += 1
            ep_iter += 1
            state = obs

            # Aktualnie hard copy / może być soft
            if iteration % Config.COPY_RATE == 0:
                target_nn.set_weights(main_nn.get_weights())
                # soft_update(target_nn, main_nn, tau=0.005)

            if (
                len(buffer) > Config.EXPLORATIONTIME
                and iteration % Config.TRAIN_EVERY == 0
            ):
                states, actions, rewards, next_states, dones = buffer.sample(
                    Config.BATCH_SIZE
                )
                loss = train_step(
                    main_nn,
                    target_nn,
                    optimizer,
                    loss_fn,
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                )

        append_results(iteration, episode, ep_real_reward, reward, loss, epsilon)

        if ep_real_reward > best:
            main_nn.save_weights(f"{RESULTS_DIR}/model_{Config.MODEL_NAME}.weights.h5")
            best = ep_real_reward
            print(
                f"[BEST] New best {best:.2f} with mod: {ep_reward}-> checkpoint saved."
            )

        avg_window.append(ep_reward)
        avg_real.append(ep_real_reward)
        if(len(buffer) > Config.EXPLORATIONTIME):
            epsilon = max(Config.EPSILON_END, epsilon * Config.EPSILON_DECAY)

        if (episode % 5) == 0:
            print(
                f"Ep {episode} | avg100={np.mean(avg_window):.2f} , {np.mean(avg_real):.2f} | "
                f"last_loss={float(loss):.4f} | eps={epsilon}"
            )
            print(
                f"Last episode {episode} rewards: {ep_reward} last episode loss: {loss} , "
                f"iter: {iteration}, best {best} best"
            )


run_training()
