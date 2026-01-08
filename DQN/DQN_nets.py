import tensorflow as tf
from tensorflow import keras

class DQN(tf.keras.Model):
    def __init__(self, action_num=5):
        super(DQN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=8, strides=4, activation="relu", padding="valid"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=4, strides=2, activation="relu", padding="valid"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation="relu", padding="valid"
        )

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(512, activation="relu")
        self.out = tf.keras.layers.Dense(action_num, dtype=tf.float32)

    def call(self, x):
        x = tf.cast(x, tf.float32)  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.out(x)

class DuelingDQN(tf.keras.Model):
    def __init__(self, action_num):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=8, strides=4, activation="relu", padding="valid"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=4, strides=2, activation="relu", padding="valid"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation="relu", padding="valid"
        )

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(512, activation="relu")

        self.V = tf.keras.layers.Dense(1)  
        self.A = tf.keras.layers.Dense(action_num)  

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)

        V = self.V(x)
        A = self.A(x)

        Q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        return Q