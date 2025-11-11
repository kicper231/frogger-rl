from collections import deque
import numpy as np
import tensorflow as tf


# https://arxiv.org/abs/1312.5602 o tym
class ReplayBuffer(object):

    def __init__(self, size, H=84, W=84, C=4):
        self.size = int(size)
        self.H, self.W, self.C = H, W, C
        self.states  = deque(maxlen=size)   
        self.actions = deque(maxlen=size)   
        self.rewards = deque(maxlen=size)  
        self.dones   = deque(maxlen=size) 
        self.next_states = deque(maxlen=size)

    def add(self, state, next_state, action, reward, done):
        self.states.append(state)
        self.actions.append(np.uint8(action))
        self.rewards.append(np.float16(reward))
        self.dones.append(bool(done))
        self.next_states.append(next_state)

    def __len__(self):
        return len(self.states)

    def _valid_indices(self, batch_size: int) -> np.ndarray:
        return np.random.choice(len(self.states) - 1, batch_size, replace=True)

    def sample(self, batch_size: int):
        idx = self._valid_indices(batch_size)

        states = np.array([self.states[i] for i in idx], dtype=np.uint8)
        next_states = np.array([self.next_states[i] for i in idx], dtype=np.uint8)

        actions = np.array([self.actions[i] for i in idx], dtype=np.int32)
        rewards = np.array([self.rewards[i] for i in idx], dtype=np.float32)
        dones = np.array([self.dones[i] for i in idx], dtype=np.float32)

        states = tf.convert_to_tensor(states, dtype=tf.uint8)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.uint8)

        states = tf.transpose(states, [0, 2, 3, 1])
        next_states = tf.transpose(next_states, [0, 2, 3, 1])

        states = tf.cast(states, tf.float32) / 255.0
        next_states = tf.cast(next_states, tf.float32) / 255.0
        
        return (
            states,
            tf.convert_to_tensor(actions),
            tf.convert_to_tensor(rewards),
            next_states,
            tf.convert_to_tensor(dones),
        )
