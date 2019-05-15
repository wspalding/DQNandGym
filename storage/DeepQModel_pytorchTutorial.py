import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import math


# Deep Q-learning model
# the nn itself is defined below, this class handles all of the data management so that the nn can act as a DQN (the network istelf just looks like a classic run of the mill nn)
class DeepQModelWrapper():
    # if we had some function Q*(state, action) -> reward,
    #   then we could find the optimal action at each timestep to maximize the reward:
    #               π∗(s)=argmax_a(Q∗(s,a))
    #                               choose a such that Q*(s,a) is max
    #
    #  π*(s) is the optimal policy
    #  this model is used to approximate Q*(state, action) -> reward
    #
    #
    def __init__(self, **kwargs):
        
        # get important kwargs
        self.input_size = kwargs.get("input_size", 4)
        self.output_size = kwargs.get("output_size", 2)
        self.learning_rate = kwargs.get("learning_rate", 0.01) # the size of the step we take when using gradient decent
        self.discount = kwargs.get("discount", 0.9) #between 0-1, makes rewards from uncertain futures worth less that near future rewards
        self.batch_size = kwargs.get("batch_size", 128)
        
        #create memory (class defined below)
        memory_size = kwargs.get("memory_size", 10000)
        self.memory = Memory(memory_size)
        
        # poilcy net is used to get decisions
        self.policy_net = dqn(self.input_size, self.output_size)
        # target net is used for training
        self.target_net = dqn(self.input_size, self.output_size)
        
        # make target_net = policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # must be called after load_state_dict
        self.target_net.eval()
        
        # set up optimizer
        optimizer_string = kwargs.get("optimizer", "RMSprop")
        self.set_optimizer(optimizer_string)

        # set up loss function
        # user huber loss
        #           δ=Q(s,a)−(r+γmaxaQ(s′,a))
        #
        #           L = 1/|B|  *  ∑ L(δ)
        #                      (s,a,s′,r) ∈ B
        #
        #
        #           L(δ)={ (1/2) * δ^2  : for |δ|≤1,
        #                { |δ| − (1/2)  : otherwise
        loss_string = kwargs.get("loss", "Huber")
        self.set_loss_function(loss_string)
    
        self.steps_done = 0
    
    
    def get_decision(self,state):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        TARGET_UPDATE = 10
        # get random # between 0-1
        sample = random.random()
        #
        episode_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1*self.steps_done/EPS_DECAY)
        self.steps_done += 1
        if sample > episode_threshold:
            with torch.no_grad():
                result = self.policy_net(state)
                # print(result, result.shape)
                # get largest colum value for each row
                result = result.max(0)
                #get second colum on max result is the index where the max element was found
                result = result[1]
                return result.view(1,1)
        else:
            # return random decision
            return torch.tensor([[random.randrange(self.output_size)]], dtype=torch.long)


    def optimize_model(self):
        #this is where we make sure the loss is computed properly, and the gradient steps are taken
        # cant take a batch size of memory if there isnt enough in the memory
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). (zip is its own inverse with *)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
#        print("batch = ", batch.next_state)

        # compute mask for non-terminal states, and concatinate the batch elements, (final states would be after the simulation ended)
        # lambda s: s is not None: returns s if s is not none
        # map(lambda s: s is not None, batch.next_state) : map applies lambda function to each in batch.next_state
        # tuple(map(lambda s: s is not None, batch.next_state)) : a tuple of all not none in batch.next_state
        # torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8) : converts the tuple to a tensor with each item as a torch.uint8
        mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
        
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        # vector of states
        state_batch = torch.cat(batch.state)
        # vector of actions
        action_batch = torch.cat(batch.action)
        # vector of rewards
        reward_batch = torch.cat(batch.reward)
        
        
        # compute Q(s_t, a) , .gather(1, action_batch): ?????
        state_action_values = self.policy_net(state_batch.reshape(-1,self.input_size)).gather(1, action_batch)
#        print("state-action values", state_action_values, "\n")
        # need to change the 4 to something based on input size
        #
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[mask] = self.target_net(non_final_next_states.reshape(-1,self.input_size)).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.discount) + reward_batch
        
        self.optimizer.zero_grad()
#        print("state_action_values: ", state_action_values)
#        print("expected_state_action_values: ", expected_state_action_values)
        loss = self.loss(state_action_values, expected_state_action_values)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    
    def set_optimizer(self, optimizer_string):
        optimizers = {
            "SGD": optim.SGD(self.policy_net.parameters(), lr=self.learning_rate),
            "RMSprop": optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate),
            }
        try:
            self.optimizer = optimizers[optimizer_string]
        except:
            self.optimizer = optimizer_string
        self.optimizer.zero_grad()

    
    def set_loss_function(self, loss_string):
        loss_functions = {
#            "MSE": nn.MSELoss(),
            "Huber": F.smooth_l1_loss,
            }
        try:
            self.loss = loss_functions[loss_string]
        except:
            self.loss = loss_string


class dqn(nn.Module):
    def __init__(self,input_shape,output_shape, **kwargs):
        super(dqn, self).__init__()
        # set up layers
        self.input_layer = nn.Linear(input_shape, 5)
        
        self.hidden1 = nn.Linear(5, 5)
        self.hidden2 = nn.Linear(5, 5)
        self.hidden3 = nn.Linear(5, 5)
        self.hidden4 = nn.Linear(5, 5)
        
        #trying to predict the the reward for left and right (1 output for left, the other for right)
        self.output_layer = nn.Linear(5,output_shape)

    def forward(self, x):
        x = x.float()
        activation1 = self.input_layer(x)
        activation2 = F.relu(self.hidden1(activation1))
        activation3 = F.relu(self.hidden2(activation2))
        activation4 = F.relu(self.hidden3(activation3))
        activation5 = F.relu(self.hidden4(activation4))
        activation6 = F.relu(self.output_layer(activation5))
        #output = torch.max(activation3,0)[1]
        # print(output)
        return activation6





# maps (state, action) pairs to (next_state, reward) pairs
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# a cyclic buffer that holds the recently observed trasitions
class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # we sample the batches randomly, so taht the transitions are "decorrelated" which has been shown to stabablize the training procedure
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
