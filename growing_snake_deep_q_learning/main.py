import numpy as np
from snake import SnakeGame
from agent import Agent
import os, time
import tensorflow as tf
from state_setting import state_growing_return
from helper import plot, get_time, write_q_values, save_figure

save_dir = '.\\saved_models' + get_time() + '\\'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

env = SnakeGame(15*40, 15*40, 999, 15 , 4, True, True, 50, state_fn=state_growing_return)

agent = Agent(n_actions=3, n_inputs=25+4+4, lr=0.0001, epsilonl=250)

# if you want to continue training previous model uncomment these lines
# agent.model = tf.keras.models.load_model('.\\saved_models23_07_20_13_00_55\\model_23_07_20_13_41_56')
# agent.epsilonl = 2

high_scores = []
mean_rewards = []
scores = []
rewards = []
epsilon = []

episodes = 200_000

s = env.reset()

for i in range(episodes):
    if i%20_000 == 0 and i != 0:
        print(1-(i/episodes), 'remaining')
        
        save_path = save_dir+'model_'+get_time()
        agent.model.save(save_path, save_format='tf')
        
        print('saved model to', save_path)
    
    a = agent(s)
    
    s_prime, reward, done, truncate = env.play_step(a)
    
    agent.train_short(s, a, reward, s_prime, done, truncate)
    
    agent.remember(s, a, reward, s_prime, done, truncate)
    
    s = s_prime
    
    rewards.append(reward)
    scores.append(env.score)
    
    if done or truncate:
        
        agent.n_games += 1
        
        agent.train_long()
        
        s = env.reset()
        
        mean_rewards.append(sum(rewards)/len(rewards))
        high_scores.append(max(scores))
        epsilon.append(max((agent.epsilon/100, 0)))
        
        scores = []
        rewards = []
        
        plot(mean_rewards, high_scores, q=epsilon, x_label='n_games',
             y_label='meanRewards', z_label='highScores', q_label='epsilon/100')

print('Done training')

save_figure(save_dir + 'results.png')

save_path = save_dir+'Qvalues_finalsave_'+get_time()

agent.model.savev(save_path, save_format='tf')