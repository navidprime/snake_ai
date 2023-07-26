import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
def plot(x, y, x_label, y_label):
    
    plt.clf()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.plot(x, y)
    plt.text(len(y)-1, y[-1], str(round(y[-1], 2)))
    
    plt.grid()
    
    plt.show(block=False)
    plt.pause(.1)

def close_window():
    plt.close()

def save_figure(path):
    plt.savefig(path)

def read_q_values(path:str):
    with open(path, 'r') as f:
        content = f.readlines()
    
    keys = []
    vals = []
    
    q_values = dict()
    
    for i in range(len(content)):
        keys.append(eval(content[i][:36].strip()))
        vals.append(np.array(eval(content[i][37:].strip())))
        
        q_values[keys[i]] = vals[i]
    
    return q_values

def write_q_values(q_values:dict, path:str):
    
    with open(path, 'a') as f:
        
        for state, actions_reward in q_values.items():
            
            f.write(f'{state}, {actions_reward.tolist()}\n')

def get_time():
    return time.strftime('%y_%m_%d_%H_%M_%S')