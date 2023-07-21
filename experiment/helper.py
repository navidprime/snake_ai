import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()
def plot(y, z, q, x_label, y_label, z_label, q_label):
    
    plt.clf()
    
    plt.xlabel(x_label)
    
    plt.plot(y, label=y_label)
    plt.text(len(y)-1, y[-1], str(round(y[-1], 2)))
    
    plt.plot(z, label=z_label)
    plt.text(len(z)-1, z[-1], str(round(z[-1], 2)))

    plt.plot(q, label=q_label)
    plt.text(len(q)-1, q[-1], str(round(q[-1], 2)))
    
    plt.legend()
    plt.grid()
    
    plt.show(block=False)
    plt.pause(.01)

def save_figure(path):
    plt.savefig(path)

def read_q_values(path:str):
    with open(path, 'r') as f:
        content = f.readlines()
    
    keys = []
    vals = []
    
    qs = dict()
    
    for i in range(len(content)):
        keys.append(eval(content[i][:36].strip()))
        vals.append(np.array(eval(content[i][37:].strip())))
        
        qs[keys[i]] = vals[i]
    
    return qs

def write_q_values(q_values:dict, path:str):
    
    with open(path, 'a') as f:
        
        for k, v in q_values.items():
            
            f.write(f'{k}, {v.tolist()}\n')

def get_time():
    return time.strftime('%y_%m_%d_%H_%M_%S')