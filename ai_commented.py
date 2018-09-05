# AI of the Car
# With Explanation

# 1. Importing libraries

import numpy as np 
# work with array

import random 
# for random samples

import os 
# to save & load model

import torch 
# neural networks

import torch.nn as nn 
# tools to implement NN

import torch.nn.functional as F 
# contains loss functions

import torch.optim as optim 
# for optimizer to perform sgd

import torch.autograd as autograd 
# for working with pytorch  

from torch.autograd import Variable 
# to create a variable with tensor with gradient


# 2. Creating Architecture of the Neural Network

class Network(nn.Module):                                                  # inheriting the module parent class | OOP
    

    def __init__(self, input_size, nb_action):                             # unitializes when class is created | self refers to the object | input_size = no of input neurons (5 that describes the state) | nb_action = 3 output actions         

        super(Network, self).__init__()                                    # SUPER to be able to use nn.Module | trick to use all tools of nn.Module
        self.input_size = input_size                                       # creating a new variable attached to the object containing no of input neurons
        self.nb_action = nb_action                                         # creating a new variable attached to the object containing output actions
        self.fc1 = nn.Linear(input_size, 30)                               # Linear for full connections | connection b/w input(5) & hidden layer(30)(by experimenting) 
        self.fc2 = nn.Linear(30, nb_action)                                # connection b/w hidden(30) and output layer(3) (possible actions(right, left, straight))
    

    def forward(self, state):                                              # self - taking variables of object | state - input for NN

        x = F.relu(self.fc1(state))                                        # x= hidden neurons | activation function rectifier for fc1 on a particular input state
        q_values = self.fc2(x)                                             # O/P neuron 
        return q_values                                                    # returning q values for each possible action


# 3. Implemention of Experience Replay


class ReplayMemory(object):                                                # for observing previous memory
    

    def __init__(self, capacity):                                          # capacity = 100000 for previous 100000 actions

        self.capacity = capacity                                           # max number of transitins in our replaymemory class
        self.memory = []                                                   # initializing list | contain last 100000 events
    

    def push(self, event):                                                 # append new transitions and take care for last 100 transitions

        self.memory.append(event)                                          # appending new events in memory | event consist of a set of 4 elements (last_state, new state, last action, last reward)
        if len(self.memory) > self.capacity:                               # condition to prohibit more then capacity (100) elements
            del self.memory[0]                                             # deleting first element
    

    def sample(self, batch_size):                                          # function to return sample | batch_size = size of sample

        samples = zip(*random.sample(self.memory, batch_size))             # random= for random samples from memory that have a fixed size of batch_size 
                                                                           # if list= ((1,2,3), (4,5,6)) with zip * it will become  ((1,4),(2,5),(3,6))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)           # convert something to convert samples to our torch_variable
                                                                           # apply lambda function to samples 
                   

# 4. Implementation of Deep Q Learning


class Dqn():
    

    def __init__(self, input_size, nb_action, gamma):                       # gamma is the dalay coefficient

        self.gamma = gamma                                                  # to attach with our object
        self.reward_window = []                                             # sliding window of the mean of the last 100 rewards to evaluate evolution of the AI
        self.model = Network(input_size, nb_action)                         # object of Network class 
        self.memory = ReplayMemory(100000) 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)    # we use Adam for SGD | lr= learning rate it should be perfect
        self.last_state = torch.Tensor(input_size).unsqueeze(0)             # vector of 5 dimensions | for pytorch it needs to bea torch tensor and also has to be a new fake dimension that corresponds to the batch
                                                                            # unsqueeze for fake dimension will be the first dimension & 0 is the index
        self.last_action = 0 
        self.last_reward = 0 
    

    def select_action(self, state):                                         # class to choose the next action | state as inputstate

        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100  # best action to play but also be exploring new ways
                                                                            # for DQL we recommend softmax then argmax since it doesnt explores much
                                                                            # convert to torch variable | volatile = True will not include the gradients in the graph computation improving performance
                                                                            # temprature is a postive number refers to surity of the decision | surity proportional to T
        action = probs.multinomial()                                        # random draws from this probability distribution
        return action.data[0,0] 
    

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):

        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)   # outputs of the batch_state
                                                                                            # gather for output of particular action (1) 
                                                                                            # batch state has the fake dimension but batch_action doesnt have so we use unsqueeze 
                                                                                            # kill fake batch with squeeze coz we want in a simple nentwork (1= index)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]                      # detach all the several models and taking max of all q values of index (1) | 0 corresponds to the state
        # see the handbook
        target = self.gamma*next_outputs + batch_reward                                      # formula for loss as per the book
        td_loss = F.smooth_l1_loss(outputs, target)                                          # loss function for our AI best loss function for DQN | argumnets re predictions and targets
        self.optimizer.zero_grad()                                                           # we reinitialize the optimizer at each iteration of loop so we use zero_grad
        td_loss.backward(retain_variables = True)                                            # for back propogation | retain_variables=true improve bp and free the memory
        self.optimizer.step()                                                                # uses optimizer to updates the weights 
    

    def update(self, reward, new_signal):

        new_state = torch.Tensor(new_signal).float().unsqueeze(0)                            # choosing new action
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)                                               # select = to play the action
        if len(self.memory.memory) > 100:                                                    # start learning when memory > 100
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)            # to learn from the action
        self.last_action = action                                                            # putting action to last action
        self.last_state = new_state                                                          # similarly above
        self.last_reward = reward                                                            # similarly above
        self.reward_window.append(reward)                                                    # similarly above
        if len(self.reward_window) > 1000:                                                   # Updating reward window 
            del self.reward_window[0]
        return action
    

    def score(self):

        return sum(self.reward_window)/(len(self.reward_window)+1.)                          # Computing Mean of rewards | +1 for avoiding error when len = 0
    

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth') 
    

    def load(self):

        if os.path.isfile('previous_AI.pth'):
            print("--> ...loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
