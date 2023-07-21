import os
import tensorflow as tf
from snake import SnakeGame
from agent_cpu import Agent
from state_setting import state_growing_return
from helper import get_time, save_figure, plot

save_dir = '.\\saved_models' + get_time() + '\\'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

env = SnakeGame(35*25, 35*25, 999, 25, 10, False, True, 33, state_fn=state_growing_return)

high_scores = []
mean_rewards = []
scores = []
rewards = []
epsilon = []

episodes = 200_000

s = env.reset()

agent = Agent(batch_size=2048, n_actions=3, 
              n_inputs=len(s), lr=.001, epsilonl=200,
              gamma=.99)

# if you want to continue training previous model uncomment these lines
# agent.model = tf.keras.models.load_model('./saved_models23_07_21_09_32_27/model_23_07_21_09_48_04')
# agent.epsilonl = 100

for i in range(episodes):
    if (i+1)%30_000 == 0:
        print('-'*20 + 'REPORT' + '-'*30)
        print((i/episodes), 'elapsed')
        print('len of agent memory: ', len(agent.memory))
        print('epsilon: ', agent.epsilon.numpy().item())
        
        save_path = save_dir+'model_'+get_time()
        agent.model.save(save_path, save_format='tf')
        print('saved model to', save_path)
        
    a = agent(s).numpy().item()
    
    s_prime, reward, done, truncate = env.play_step(a)
    
    agent.remember_and_train_short(s, a, reward, s_prime, done, truncate)
    
    s = s_prime
    
    rewards.append(reward)
    scores.append(env.score)
    
    if done or truncate:
        
        agent.when_episode_done()
        agent.train_long()
        
        s = env.reset()
        
        mean_rewards.append(sum(rewards)/len(rewards))
        high_scores.append(max(scores))
        epsilon.append(max((agent.epsilon.numpy().item()/100, 0)))
        
        scores = []
        rewards = []
        
        plot(mean_rewards, high_scores, q=epsilon, x_label='n_games',
             y_label='meanRewards', z_label='highScores', q_label='epsilon/100')

print('Done training')

save_figure(save_dir + 'results.png')

save_path = save_dir+'Qvalues_finalsave_'+get_time()

agent.model.save(save_path, save_format='tf')