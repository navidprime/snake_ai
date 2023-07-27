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

def get_time():
    return time.strftime('%y_%m_%d_%H_%M_%S')