import gym
import numpy as np
import torch
from DeepQModel import DQNwrapper
from storage.q_learning import Q_model

#from gym import envs
#print(envs.registry.all())

env = gym.make('Breakout-ram-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('CartPole-v0')
#env = gym.make('FrozenLake8x8-v0')


print("action space size: ", env.action_space.n)
print("observation space shape: ", env.observation_space.shape)
#print("observation space lower bound: ", env.observation_space.low)
#print("observation space upper bound: ", env.observation_space.high)


model = DQNwrapper(env.observation_space.shape[0], env.action_space.n)
#model = Q_model(env)
# model.train()


epochs = 500
#target_update = 50
max_score = 0
best_round = 0
for i in range(0, epochs):
    count = 0
    observation = env.reset()
    done = False
    while not done:
        env.render() # returns the screen
        count += 1
        
        action = model.get_action(torch.tensor(observation))
        #action = model.get_action(observation)
        
        #print(observation)
#        old_obs = observation
        next_observation, reward, done, info = env.step(action)
        reward = reward if not done else -10
        model.remember(observation, action, reward, next_observation, done)
        
        observation = next_observation

        if done:
            break

    model.replay()
    if count > max_score:
        max_score = count
        best_round = i
        print(i, ": ", count, ", e: {:.2}, NEW BEST!".format(model.epsilon))
    else:
        print(i, ": ", count, ", e: {:.2}".format(model.epsilon))
print("Best score: ", max_score, " on round: ", best_round)
env.close()
