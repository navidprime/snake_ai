import numpy as np
import random

class Agent:
    
    def __init__(self, n_actions=None, lr=0.1, gamma=.99, epsilon_length=200) -> None:
        self.lr = lr
        self.gamma = gamma
        self.epsilon_length = epsilon_length
        self.n_actions = n_actions
        self.epsilon = 0
        self.n_rand_preds = 0
        
        
        self.epsilon = 0
        self.test_mode = False
        
        self.q_values = dict()
    
    def __get_random_array(self):
        return np.random.normal(loc=0, scale=.1, size=(3,)).astype('float32')
    
    def predict(self, state):
        random_choice_proba = random.randint(0, self.epsilon_length)
        
        if (not self.epsilon > random_choice_proba) and (not self.test_mode):
            self.n_rand_preds += 1
            return np.random.randint(0, self.n_actions)
        else:
            state = tuple(state)
            
            if state in self.q_values.keys():
                return np.argmax(self.q_values[state])
            else:
                self.n_rand_preds += 1
                self.q_values[state] = self.__get_random_array()

                return np.random.randint(0, self.n_actions)
    
    def __call__(self, state):
        return self.predict(state)
    
    def update_q_values(self, state, action, new_state, reward):
        state = tuple(state)
        new_state = tuple(new_state)
        
        if not state in self.q_values.keys():
            self.q_values[state] = self.__get_random_array()
            
        if not new_state in self.q_values.keys():
            self.q_values[new_state] = self.__get_random_array()
        
        self.q_values[state][action] = self.q_values[state][action] \
            + (self.lr * (reward + (self.gamma * np.max(self.q_values[new_state])) - self.q_values[state][action]))