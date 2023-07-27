import os
import tensorflow as tf
from snake import SnakeGame
from agent_gpu import Agent
from state_setting import state_growing_return
from helper import *

save_dir = './saved_models_' + get_time()[:8] + '/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

width = 40*20
height = 40*20
step = 20
fps_on_train = 360
fps_on_test = 30
truncate_timeout = 200
lr_schduler = lambda n_games, lr: lr if n_games < 300 else lr*tf.exp(-0.01)

def train():
    global model_save_path
    
    env = SnakeGame(width, height, fps_on_train, step, 4, True, True, truncate_timeout, state_fn=state_growing_return)
    
    s = env.reset()
    
    print(f'len state: {len(s)}, {s}')
    agent = Agent(3, len(s), 16, lr_schduler, lr=.002, epsilon_length=350)
    
    scores_each_episode = []
    
    episodes = 300_000
    
    for episode in range(episodes):
        if episode % 1000 == 0 and episode != 0:
            print('-'*50)
            
            print(f'episode: {episode}')
            
            # print(f'memory capacity: {len(agent.memory)}')
            
            print(f'lr: {agent.optimizer.learning_rate}')
            
            print(f'n_games: {agent.n_games}')
            
            print(f'epsilon: {agent.epsilon.numpy().item()}, epsilon_length: {agent.epsilonl}')
        
        a = agent(s).numpy().item()
        
        s_, reward, done, truncate = env.play_step(a)
        
        agent.remember_and_train_short(s, a, reward, s_, done, truncate)
        
        s = s_
        
        if done or truncate:
            
            agent.when_episode_done()
            agent.train_long()
            
            scores_each_episode.append(env.score)
            
            plot(range(len(scores_each_episode)), scores_each_episode, 'n_games', 'scores')
            
            s = env.reset()
        
    print('Done training.')
    
    save_figure(save_dir + 'results.png')
    print('saved figure')

    model_save_path = save_dir+'saved_model_'+get_time()

    agent.model.save(model_save_path, save_format='tf')
    print('saved model.')

def test():
    global model_save_path

    print('Testing the agent')
    
    env = SnakeGame(width, height, fps_on_test, step , 4, True, True, truncate_timeout, state_fn=state_growing_return)
    
    s = env.reset()
    
    agent = Agent(3, len(s), 1, lambda n, lr:lr)
    agent.model = tf.keras.models.load_model(model_save_path)
    agent.test_mode = True
    
    episodes = 10_000
    
    for i in range(episodes):
        a = agent(s)
        s_prime, reward, done, truncate = env.play_step(a)
        
        s = s_prime
        
        if done or truncate:
            s = env.reset()

if __name__ == '__main__':
    # train()
    model_save_path = './saved_models_23_07_27/saved_model_23_07_27_11_17_33'
    test()