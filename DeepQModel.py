import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import math

class DQNwrapper:
    def __init__(self, state_size, action_size, **kwargs):
        #required parameters
        self.state_size = state_size
        self.action_size = action_size

        # hyper paramerters
        mem_size = kwargs.get("memory_size", 2000)
        self.gamma = kwargs.get("discount_factor", 0.99)
        # 1 = 100% exploration
        self.epsilon = kwargs.get("exploration_rate", 1.0)
        self.epsilon_decay = kwargs.get("exploration_decay", 0.99)
        self.epsilon_min = kwargs.get("exploration_min", 0.01)
        
        self.learning_rate = kwargs.get("leaning_rate", 0.001)
        self.batch_size = kwargs.get("batch_size", 32)
        

        self.memory = deque(maxlen=mem_size)
        self.model = kwargs.get("model", dqn(state_size,action_size))
#        assert(self.model.input_shape == self.state_size)
#        assert(self.model.output_shape == self.action_size)
        self.set_optimizer(kwargs.get("optimizer", "RMSprop"))
        self.set_loss_function(kwargs.get("loss", "Huber"))
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # exploring: get random action
            action = random.randrange(self.action_size)
#            print("random action: ", action)
            return action
        action_values = self.model.forward(state).data.numpy()
        action = np.argmax(action_values)
#        print("model action: ", action_values)
        return action
            
    def replay(self):
        if self.batch_size > len(self.memory):
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
                    # no need to predict future rewards
            else:
                n_state = torch.tensor(next_state)
#                print(n_state)
                pred = self.model.forward(n_state).data.numpy()
#                print("pred = ", pred)
                target = (reward + self.gamma * np.amax(pred))
#                print("max pred = ", np.amax(pred))

            future_target = self.model.forward(torch.tensor(state))
#            print("pre: ", future_target)
            future_target[action] = target
#            print("post: ", future_target, "<-", target, "\n")

            self.optimizer.zero_grad()
            x = torch.tensor(state).float()
            y = future_target.unsqueeze(1)
#            print("state = ", x, x.shape)
#            print("future target = ", y, y.shape, "\n")
            loss = self.loss(x, y)
            loss.backward()
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_optimizer(self, optimizer_string):
        optimizers = {
            "SGD": optim.SGD(self.model.parameters(), lr=self.learning_rate),
            "RMSprop": optim.RMSprop(self.model.parameters(), lr=self.learning_rate),
            }
        self.optimizer = optimizers.get(optimizer_string, optimizer_string)
#        try:
#            self.optimizer = optimizers[optimizer_string]
#        except:
#            self.optimizer = optimizer_string
        self.optimizer.zero_grad()


    def set_loss_function(self, loss_string):
        loss_functions = {
            "MSE": F.mse_loss,
            "Huber": F.smooth_l1_loss,
        }
        self.loss = loss_functions.get(loss_string, loss_string)
#        try:
#            self.loss = loss_functions[loss_string]
#        except:
#            self.loss = loss_string


class dqn(nn.Module):
    def __init__(self,input_shape,output_shape, **kwargs):
        super(dqn, self).__init__()
        # set up layers
        self.input_layer = nn.Linear(input_shape, 24)
        self.hidden1 = nn.Linear(24, 24)
#        self.hidden2 = nn.Linear(24, 40)
#        self.hidden3 = nn.Linear(40, 20)
#        self.hidden4 = nn.Linear(20, 5)
        #trying to predict the the reward for left and right (1 output for left, the other for right)
        self.output_layer = nn.Linear(24,output_shape)

    def forward(self, x):
        x = x.float()
        activation1 = F.relu(self.input_layer(x))
        activation2 = F.relu(self.hidden1(activation1))
#        activation3 = F.relu(self.hidden2(activation2))
#        activation4 = F.relu(self.hidden3(activation3))
#        activation5 = F.relu(self.hidden4(activation4))
        activation6 = self.output_layer(activation2)
        #output = torch.max(activation3,0)[1]
        # print(output)
        return activation6




