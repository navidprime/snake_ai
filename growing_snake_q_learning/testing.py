from snake import SnakeGame
from state_setting import state_growing_return
from agent import Agent
from helper import read_q_values

# i've removed unncessary files on the saved q values
trained_q_values = read_q_values('Q_values_23_07_19_20_55_15\\Qvalues_finalsave_23_07_19_21_10_46.csv')
env = SnakeGame(25*25, 25*20, 30, 25 , 1, True, True, state_fn=state_growing_return)

agent = Agent(None, n_actions=3)
agent.table = trained_q_values
agent.test_mode = True

episodes = 10_000

s = env.reset()

for i in range(episodes):
    a = agent(s)
    s_prime, reward, done, truncate = env.play_step(a)
    
    s = s_prime
    
    if done or truncate:
        s = env.reset()