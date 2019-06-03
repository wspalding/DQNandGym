import gym
import numpy as np
import torch
from DeepQModel import DQNwrapper
from storage.q_learning import Q_model
import networks

#from gym import envs
#print(envs.registry.all())

env_name = 'CartPole-v0'
#env_name = 'Breakout-ram-v0'
#env_name = 'Acrobot-v1'
#env_name = 'FrozenLake8x8-v0'

env = gym.make(env_name)


model_dir = 'model_output/' + env_name + '/'


print("action space size: ", env.action_space.n)
print("observation space shape: ", env.observation_space.shape)
#print("observation space lower bound: ", env.observation_space.low)
#print("observation space upper bound: ", env.observation_space.high)

network = networks.cartpole_RNN_dqn(env.observation_space.shape[0], env.action_space.n)
model = DQNwrapper(env.observation_space.shape[0], env.action_space.n, model=network)
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
        # save model
#        model_file = 'cartpole_model_score_{}'.format(i)
#        torch.save(model.model.state_dict(), model_dir + model_file)
    else:
        print(i, ": ", count, ", e: {:.2}".format(model.epsilon))
print("Best score: ", max_score, " on round: ", best_round)
env.close()
