from agent_cpu import Agent
import gymnasium as gym
import numpy as np
import tensorflow as tf

# adam w -> i : 7048, max 359, n_games 233
# adam w -> i : 8283, max 500, n_games 251
# adam w with xiaer init: learned faster. i : 6777, max 443, n_games: 220
# adam w with xiaer with huber loss fn : didn't learn
# adam -> ~ 8736, max 282, n_games 249
# adak with xiaer init: it didn't go further than 200 but it learned something from 6500 (propably it needed more time)
# adamax -> reached 80 but decayed
# rmsprop -> converges slower
# adadelte -> didn't learn at all
# sgd with momentom -> no
# Adagard -> no

env = gym.make('CartPole-v1')

s = env.reset()[0]

agent = Agent(2, len(s), batch_size=2048, lr_schduler=lambda x,y: y if x < 175 else y*tf.exp(-.01), lr=.002)

old_i = 0

for i in range(10000):
    
    if (i+1)%200 == 0:
        print(i)
        print(agent.optimizer.learning_rate)
        print(agent.epsilon)
        print(len(agent.memory))
    
    a = agent(s).numpy().item()
    
    s_, r, d, t, *_ = env.step(a)
    # s_ = np.array([s_])
    
    # print(s, a, r, s_, d, t)
    agent.remember_and_train_short(s, a, r, s_, d, t)
    
    s = s_
    
    if d or t:
        agent.when_episode_done()
        agent.train_long()
        
        s = env.reset()[0]
        
        print('kept poll for: ',i - old_i)
        if i - old_i > 200:
            print('at i: ', i, 'n_games: ', agent.n_games.numpy().item())
            break
        
        old_i = i