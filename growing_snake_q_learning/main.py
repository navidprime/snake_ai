import os
import numpy as np
from snake import SnakeGame
from agent import Agent
from state_setting import state_growing_return
from helper import *

width = 60*20
height = 40*20
step = 20
fps_on_train = 360
fps_on_test = 30
truncate_timeout = 150

save_dir = './Q_values_' + get_time()[:8] + '/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train():
    global q_values_path
    
    env = SnakeGame(width, height, fps_on_train, step , 1, True, True, truncate_timeout, state_fn=state_growing_return)

    agent = Agent(n_actions=3, epsilon_length=200, lr=.01)

    scores_each_episode = []

    episodes = 200_000

    s = env.reset()
    
    for episode in range(episodes):
        if episode % 1000 == 0 and episode != 0:
            print('-'*50)
            
            print(f'episode: {episode}')
            
            print(f'length of states: {len(agent.q_values)}')
            
            print(f'n_rand_preds: {agent.n_rand_preds}')
            
            print(f'epsilon: {agent.epsilon}, epsilon_length: {agent.epsilon_length}')
        
        a = agent(s)
        
        s_, r, done, truncate = env.play_step(a)
        
        agent.update_q_values(s, a, s_, r)
        
        s = s_
        
        if done or truncate:
            scores_each_episode.append(env.score)
            
            agent.epsilon += 1
            
            plot(np.arange(0, len(scores_each_episode)), scores_each_episode, 'n_games', 'high_score')
            
            s = env.reset()
            
    print('Done training.')
    
    q_values_path = save_dir+'qvalues_'+get_time()+'.csv'
    
    write_q_values(agent.q_values, q_values_path)
    print('wrote q values.')
    
    save_figure(save_dir + 'results.png')
    close_window()

def test():
    global q_values_path
    
    print('Testing the agent')
    
    env = SnakeGame(width, height, fps_on_test, step , 1, True, True, truncate_timeout, state_fn=state_growing_return)

    agent = Agent(3)
    agent.test_mode = True
    
    agent.q_values = read_q_values(q_values_path)
    
    s = env.reset()
    
    episodes = 2_000
    
    highest_score = 0

    for i in range(episodes):
        a = agent(s)
        
        s_prime, reward, done, truncate = env.play_step(a)
        
        s = s_prime
        
        if done or truncate:
            if env.score > highest_score:
                highest_score = env.score
                
            s = env.reset()
    
    print(f'highest_score: {highest_score}')

if __name__ == '__main__':
    train()
    test()