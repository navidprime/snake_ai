import numpy as np
from snake import SnakeGame
from agent import Agent
import os
from state_setting import state_no_growing_return
from helper import plot, get_time, write_q_values, save_figure

save_dir = '.\\Q_values_' + get_time() + '\\'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

env = SnakeGame(50*12, 50*12, 240, 50 , 1, True, False, 75, state_fn=state_no_growing_return)

agent = Agent(None, n_actions=3, lr=0.001)

high_scores = []
mean_rewards = []
scores = []
rewards = []
diff_states = []

episodes = 100_000

s = env.reset()

for i in range(episodes):
    if i%20_000 == 0 and i != 0:
        print(1-(i/episodes), 'remaining')
        
        save_path = save_dir+'Qvalues_'+get_time()+'.csv'
        write_q_values(agent.table, save_path)
        
        print('saved q values to ', save_path)
        
    a = agent(s)
    
    s_prime, reward, done, truncate = env.play_step(a)
    
    agent.update_table(s, a, s_prime, reward)
    
    s = s_prime
    
    rewards.append(reward)
    scores.append(env.score)
    
    if done or truncate:
        mean_rewards.append(sum(rewards)/len(rewards))
        high_scores.append(max(scores))
        diff_states.append(len(agent.table)/10)
        
        scores = []
        rewards = []
        
        plot(mean_rewards, high_scores, diff_states, x_label='n_games',
             y_label='meanRewards', z_label='highScores', q_label='diffStates/10')
        
        s = env.reset()
    
    
print('Done training')

save_figure(save_dir + 'results.png')

save_path = save_dir+'Qvalues_'+get_time()+'_finalsave'+'.csv'

write_q_values(agent.table, save_path)