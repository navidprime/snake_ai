import numpy as np
import tensorflow as tf
from collections import deque
import random
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

MAX_MEMMORY = 100_000

class Model(tf.keras.Model):
    def __init__(self, n_outputs, n_inputs, units=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hidden1 = tf.keras.layers.Dense(units, activation='relu', input_shape=(n_inputs,))
        self.output_ = tf.keras.layers.Dense(n_outputs)
    
    def call(self, input_):
        assert len(input_.shape) == 2
          
        x = self.hidden1(input_)
        x = self.output_(x)
        
        return x
        
class Agent:
    
    def __init__(self, n_actions=3, n_inputs=25+4+4, batch_size=1024, lr=0.01, gamma=.99, epsilonl=100) -> None:
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 0
        self.epsilonl = epsilonl
        self.n_games = 0
        self.n_actions = n_actions
        self.test_mode = False
        
        self.model = Model(n_outputs=n_actions, n_inputs=n_inputs)
        self.optimizer = tf.optimizers.Adam(lr)
        self.loss_fn = tf.losses.mean_squared_error
        
        self.memmory = deque(maxlen=MAX_MEMMORY)
    
    def predict(self, state):
        state = np.array([state])
        
        proba = np.random.randint(0, self.epsilonl)
        
        self.epsilon = self.epsilonl - self.n_games
        
        if proba < self.epsilon and not self.test_mode:
            return np.random.randint(0, self.n_actions)
        else:
            return tf.argmax(self.model(state), axis=1).numpy().item()
    
    def __call__(self, state):
        return self.predict(state)

    def remember(self, state, action, reward, next_state, done, truncate):
        self.memmory.append((state, action, reward, next_state, done, truncate))
    
    def train_step(self, states, actions, rewards, next_states, dones, truncates):
        assert type(states) == np.ndarray
        assert len(states.shape) == 2
        
        next_q_values = self.model(next_states)
        
        runs = 1.0 - (truncates | dones)
        
        target = rewards + runs * self.gamma * tf.reduce_max(next_q_values, axis=1)
        
        target = target.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.n_actions)
        
        with tf.GradientTape() as tape:
            
            all_q_values = self.model(states)
            
            Q_values = tf.reduce_sum(all_q_values * mask, axis=-1, keepdims=True)
            
            loss = tf.reduce_mean(self.loss_fn(target, Q_values))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    
    def train_short(self, state, action, reward, next_state, done, truncate):
        state = np.array([state])
        action = np.array([[action]])
        reward = np.array([[reward]])
        next_state = np.array([next_state])
        done = np.array([[done]])
        truncate = np.array([[truncate]])
        
        self.train_step(state, action, reward, next_state, done, truncate)
    
    def train_long(self):
        if len(self.memmory) > self.batch_size:
            mini_sample = random.sample(self.memmory, self.batch_size)
        else:
            mini_sample = self.memmory
        
        states, actions, rewards, next_states, dones, truncates = zip(*mini_sample)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        truncates = np.array(truncates)
        
        self.train_step(states, actions, rewards, next_states, dones, truncates)