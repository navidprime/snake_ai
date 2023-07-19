from snake import SnakeGame
from state_setting import state_no_growing_return
from agent import Agent
from helper import read_q_values

# i have stopped the training because the agent had passed 1000 score!
trained_q_values = read_q_values('Q_values_23_07_19_21_21_02\\Qvalues_23_07_19_21_22_37.csv')
env = SnakeGame(25*26, 25*20, 240, 50 , 1, True, False, state_fn=state_no_growing_return)

agent = Agent(None, n_actions=3)
agent.table = trained_q_values
agent.test_mode = True

episodes = 100_000

s = env.reset()

for i in range(episodes):
    a = agent(s)
    s_prime, reward, done, truncate = env.play_step(a)
    
    s = s_prime
    
    if done or truncate:
        s = env.reset()