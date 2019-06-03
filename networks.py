import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import math


class cartpole_linear_dqn(nn.Module):
    def __init__(self,input_shape,output_shape, **kwargs):
        super(cartpole_dqn, self).__init__()
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



class cartpole_RNN_dqn(nn.Module):
    def __init__(self,input_shape,output_shape, **kwargs):
        super(cartpole_RNN_dqn, self).__init__()
        # set up layers
        # parameters for RNN are (input_size, hidden_size, num layers)
        self.input_layer = nn.RNN(input_shape, 24, 1, nonlinearity='relu')
        self.layer1 = nn.RNN(24, 48, 1, nonlinearity='relu')
        self.layer2 = nn.RNN(48, 24, 1, nonlinearity='relu')
        self.output_layer = nn.Linear(24, output_shape, 1)
        
    
    def forward(self, x):
        x = x.float()
        a1, h1 = self.input_layer(x)
        a2, h2 = self.layer1(a1,h1)
        a3, h3 = self.layer2(a2,h2)
        out = self.output_layer(a3)
        return out
