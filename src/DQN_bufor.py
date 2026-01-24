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
        
class NStepReplayBuffer(object):

    def __init__(self, size, n_step=1, gamma=0.99, H=84, W=84, C=4):
        self.size = int(size)
        self.H, self.W, self.C = H, W, C
        self.n_step = n_step
        self.gamma = gamma
        
        # Main replay buffer
        self.states  = deque(maxlen=size)   
        self.actions = deque(maxlen=size)   
        self.rewards = deque(maxlen=size)
        self.dones   = deque(maxlen=size) 
        self.next_states = deque(maxlen=size)
        
        # Temporary n-step buffer
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        # R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^{n-1}*r_{t+n-1}
        reward = 0.0
        for i, transition in enumerate(self.n_step_buffer):
            state, next_state, action, r, done = transition
            reward += (self.gamma ** i) * r
            if done:
                return self.n_step_buffer[0][0], next_state, self.n_step_buffer[0][2], reward, done
        
        return (
            self.n_step_buffer[0][0],  # s_t
            self.n_step_buffer[-1][1], # s_{t+n}
            self.n_step_buffer[0][2],  # a_t
            reward,                    # n-step return
            self.n_step_buffer[-1][4]  # done_{t+n}
        )

    def add(self, state, next_state, action, reward, done):
        transition = (state, next_state, action, reward, done)
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) == self.n_step or done:
            state_n, next_state_n, action_n, reward_n, done_n = self._get_n_step_info()
            self.states.append(state_n)
            self.actions.append(np.uint8(action_n))
            self.rewards.append(np.float32(reward_n))
            self.dones.append(bool(done_n))
            self.next_states.append(next_state_n)
        
        if done:
            self.n_step_buffer.clear()

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


class PrioritiesReplayBuffer(object):

    def __init__(self, size, H=84, W=84, C=4):
        self.size = int(size)
        self.H, self.W, self.C = H, W, C
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.dones = deque(maxlen=size)
        self.next_states = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    def add(self, state, next_state, action, reward, done, priority):
        self.states.append(state)
        self.actions.append(np.uint8(action))
        self.rewards.append(np.float16(reward))
        self.dones.append(bool(done))
        self.next_states.append(next_state)
        self.priorities.append(1)

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
