from snake import SnakeGame
from state_setting import state_growing_return
from agent import Agent
import tensorflow as tf

# i've removed unncessary files on the saved q values
env = SnakeGame(25*25, 25*25, 60, 25, 3, True, True, 50, state_fn=state_growing_return)

agent = Agent(n_actions=3)
agent.test_mode = True
agent.model = tf.keras.models.load_model('./saved_models23_07_20_13_00_55/model_23_07_20_13_41_56')

episodes = 10_000

s = env.reset()

for i in range(episodes):
    a = agent(s)
    s_prime, reward, done, truncate = env.play_step(a)
    
    s = s_prime
    
    if done or truncate:
        s = env.reset()