import numpy as np

class Agent:
    
    def __init__(self, table, n_actions=None, lr=0.1, gamma=.99, epsilon=.99, epsilon_decay=.001) -> None:
        self.lr = lr
        self.gamma = gamma
        self.epsilon=epsilon
        self.epsilon_decay = epsilon_decay
        self.n_rand_preds = 0
        self.test_mode = False

        self.table = table
        
        if type(table) == np.ndarray:
            self.n_actions = table.shape[-1]
        else:
            self.table = dict()
            self.n_actions = n_actions
    
    def predict(self, state):
        proba = np.random.random()
        state = tuple(state)
        
        if self.epsilon > proba and not self.test_mode:
            self.epsilon -= self.epsilon_decay
            self.n_rand_preds += 1
            return np.random.randint(0, self.n_actions if type(self.table) == dict else self.table.shape[-1])
        else:
            if type(self.table.get(state)) == np.ndarray:
                return np.argmax(self.table[state])
            else:
                self.table[state] = np.zeros((self.n_actions))
                self.n_rand_preds += 1
                return np.random.randint(0, self.n_actions)
    
    def __call__(self, state):
        return self.predict(state)
    
    def update_table(self, state, action, new_state, reward):
        
        if type(self.table) == np.ndarray:
            self.table[state, action] = self.table[state, action]\
+ (self.lr * (reward + (self.gamma * np.max(self.table[new_state])) - self.table[state, action]))
        else:
            state = tuple(state)
            new_state = tuple(new_state)
            
            if not type(self.table.get(state)) == np.ndarray:
                self.table[state] = np.zeros((self.n_actions))
                
            if not type(self.table.get(new_state)) == np.ndarray:
                self.table[new_state] = np.zeros((self.n_actions))
                
            self.table[state][action] = self.table[state][action]\
                + (self.lr * (reward + (self.gamma * np.max(self.table[new_state])) - self.table[state][action]))
            