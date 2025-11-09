import gymnasium as gym
from stable_baselines3 import PPO
import ale_py
from base_parameters import Config
import tensorflow as tf
from tensorflow import keras
import numpy as np
from buffor import ReplayBuffer

class DQN(tf.keras.Model):

    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(
            5, dtype=tf.float32)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

@tf.function
def train_step(states, actions, rewards, next_states, dones):

    next_qs = target_nn(next_states)
    max_next_qs = tf.reduce_max(next_qs, axis=-1)
    # bellman equation
    # L(0) = E((r + y* maxQ(s',a',0')  - Q(s,a,0))^2)
    target = rewards + (1.0 - dones) * Config.DISCOUNT * max_next_qs

    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, Config.ACTION_NUM)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)

    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss


gym.register_envs(ale_py)
env = gym.make("ALE/Frogger-v5", obs_type="ram")
main_nn = DQN()
target_nn = DQN()
optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()
iter = 1
buffor = ReplayBuffer(100000)
epsilon = Config.EPSILON

def select_greedy(state, epsilon = Config.EPSILON):
    result = tf.random.uniform((1,))
    if result < epsilon:
        return env.action_space.sample()
    else: 
        result = main_nn.predict(state, verbose=0)
        action = np.argmax(result)
        return action


for episodes in range(Config.NUM_EPISODES + 1):
    state, info = env.reset()
    state = np.expand_dims(state, axis=0) / 255.0
    state = np.expand_dims(state, axis=0)
    done = False
    loss = 0
    ep_reward = 0

    while not done:

        # prediction
        action = select_greedy(state,epsilon=epsilon)

        # action
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.expand_dims(obs, axis=0) / 255.0

        if(action == 1):
            reward += 0.1

        if (action == 0):
            reward -= 0.1

        obs = np.expand_dims(obs, axis=0)
        done = terminated or truncated

        # buffor
        buffor.add(state, action, reward, obs, done)
        ep_reward += reward

        # # copy weights
        # if iter % Config.COPY_RATE == 0:
        #     )

        iter += 1
        state = obs

        # train main
        if(len(buffor) > Config.BATCH_SIZE):
            states, actions, rewards, next_states, dones = buffor.sample(Config.BATCH_SIZE)
            loss = train_step(states, actions, rewards, next_states, dones)

    epsilon = max(Config.EPSILON_END, epsilon * Config.EPSILON)
    target_nn.set_weights(main_nn.get_weights())    
    print(f'Last episode rewards: {ep_reward} last episode loss: {loss} \n')

print(iter)
main_nn.save_weights("dqn_weights.h5")


test_env = gym.make("ALE/Frogger-v5", render_mode="human", obs_type="ram")
obs, info = test_env.reset()
done = False


while not done:
    obs = np.expand_dims(obs, axis=0)
    result = main_nn.predict(obs, verbose=0)
    action = np.argmax(result)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    test_env.render()
