import gymnasium as gym
import ale_py
from base_parameters import Config
import tensorflow as tf
from tensorflow import keras
import numpy as np
from buffor import ReplayBuffer
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
from collections import deque
from model_DQN import DQN, DuelingDQN

@tf.function
def train_step(states, actions, rewards, next_states, dones):
    # double dqn
    next_q_main = main_nn(next_states, training=False)
    next_actions = tf.argmax(next_q_main, axis=1)

    next_q_target = target_nn(next_states, training=False)
    next_action_masks = tf.one_hot(next_actions, Config.ACTION_NUM)
    max_next_qs = tf.reduce_sum(next_action_masks * next_q_target, axis=1)

    # bellman equation
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


def soft_update(target, online, tau=0.003):
    target_weights = target.get_weights()
    online_weights = online.get_weights()

    new_weights = []
    for tw, ow in zip(target_weights, online_weights):
        new_weights.append(tau * ow + (1 - tau) * tw)

    target.set_weights(new_weights)


gym.register_envs(ale_py)

env = gym.make("ALE/Frogger-v5", obs_type="grayscale", frameskip=1)

env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=8)

env = FrameStackObservation(env, stack_size=4)

tf.keras.backend.clear_session()

main_nn = DuelingDQN(5)
target_nn = DuelingDQN(5)
zeros = tf.zeros((1, 84, 84, 4), dtype=tf.float32)
main_nn(zeros)
target_nn(zeros)
target_nn.set_weights(main_nn.get_weights())
optimizer = tf.keras.optimizers.AdamW(learning_rate=Config.LEARNING_RATE, epsilon=1e-5)
loss_fn = tf.keras.losses.Huber(delta=1.0)
iter = 1
buffor = ReplayBuffer(Config.BUFFERSIZE)
epsilon = Config.EPSILON
avg_window = deque(maxlen=100)
avg_real = deque(maxlen=100)
best = 0
save_que = []

def select_greedy(state, epsilon=Config.EPSILON):
    if  tf.random.uniform(()) < epsilon:
        return env.action_space.sample()
    else:
        # 1 84 84 4
        state = tf.transpose(state, perm=[1, 2, 0])
        state = tf.convert_to_tensor(state, dtype=tf.uint8)
        state = tf.expand_dims(state, axis=0)
        state = tf.cast(state, tf.float32) / 255.0

        result = main_nn(state, training=False)
        return int(tf.argmax(result[0]).numpy())

with open(f"results_{Config.MODEL_NAME}.txt", "a") as f:
        f.write(f"iter, episode, ep_real_reward, reward, loss, epsilon \n")

for episode in range(Config.NUM_EPISODES + 1):

    state, info = env.reset()
    done = False
    loss = 0
    ep_reward = 0
    ep_iter = 0
    lifes = 4
    ep_real_reward = 0
    iters = 0

    while not done:

        # prediction
        action = select_greedy(state, epsilon)
        # print(action)
        if ep_iter < 30:
            obs, reward, terminated, truncated, info = env.step(0)
            ep_iter += 1
            continue
        action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # reward = np.sign(reward)

        ep_real_reward += reward
        # if ep_iter < 70:
        #     ep_iter += 1
        #     continue

        if lifes != info['lives']:
            lifes-=1
            reward-=10
        # corrections
        if (action == 4 ):
            reward -= 1

        if(action == 0):
            reward -=0.5
        # if action == 1:
        #     reward += 1

        # if reward > 0:
        #     reward *= 2
        # reward += 0

        buffor.add(state, obs, action, reward, done)
        ep_reward += reward

        iter += 1
        ep_iter +=1
        state = obs

        # if iter % Config.COPY_RATE == 0:
        #     target_nn.set_weights(main_nn.get_weights())

        if len(buffor) > Config.EXPLORATIONTIME:
            epsilon = max(Config.EPSILON_END, epsilon * Config.EPSILON_DECAY)

        if iter % Config.COPY_RATE == 0:
            target_nn.set_weights(main_nn.get_weights())

        # train main
        if len(buffor) > Config.EXPLORATIONTIME and iter % Config.TRAIN_EVERY == 0:
            states, actions, rewards, next_states, dones = buffor.sample(
                Config.BATCH_SIZE
            )
            loss = train_step(states, actions, rewards, next_states, dones)
            # soft_update(target_nn, main_nn, tau=0.005)

    with open(f"results_{Config.MODEL_NAME}.txt", "a") as f:
        f.write(f"{iter}, {episode}, {ep_real_reward}, {reward}, {loss}, {epsilon} \n")

    if ep_real_reward > best:
        main_nn.save_weights(f"model_{Config.MODEL_NAME}.h5")
        best = ep_real_reward
        print(f"[BEST] New best {best:.2f} with mod: {ep_reward}-> checkpoint saved.")

    avg_window.append(ep_reward)
    avg_real.append(ep_real_reward)

    # TODO test

    if (episode % 5) == 0:
        print(
            f"Ep {episode} | avg100={np.mean(avg_window):.2f} , {np.mean(avg_real):.2f} | last_loss={float(loss):.4f} | eps={epsilon}"
        )
        print(
            f"Last episode {episode} rewards: {ep_reward} last episode loss: {loss} , iter: {iter}, best {best} best"
        )
