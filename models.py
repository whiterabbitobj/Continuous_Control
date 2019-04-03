# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



WEIGHT_LOW = -3e-2
WEIGHT_HIGH = 3e-2



def initialize_weights(net, low, high):
    for param in net.parameters():
        param.data.uniform_(low, high)



class ActorNet(nn.Module):
    """
    Actor network that approximates the non-linear function π(θ)
    """
    def __init__(self,
                 layer_sizes,
                 state_size,
                 action_size,
                 weight_low = WEIGHT_LOW,
                 weight_high = WEIGHT_HIGH):
        super(ActorNet, self).__init__()

        #currently errors if user were to provide a custom layer_sizes array
        #with dimensions other than 2x1
        fc1, fc2 = layer_sizes

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.output = nn.Linear(fc2, action_size)
        initialize_weights(self, weight_low, weight_high)

    def forward(self, state):
        """
        Build a network that maps states -> actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.output(x)
        action = logits.tanh()
        return action



class CriticNet(nn.Module):
    """
    Critic network that approximates the Value of the suggested Actions that
    come from the Actor network.

    Utilizes the actions as an input to the second hidden layer in order to
    approximate the continuous control problem.
    """

    def __init__(self,
                 layer_sizes,
                 state_size,
                 action_size,
                 num_atoms,
                 weight_low = WEIGHT_LOW,
                 weight_high = WEIGHT_HIGH):
        super(CriticNet, self).__init__()

        #currently errors if user were to provide a custom layer_sizes array
        #with dimensions other than 2x1
        fc1, fc2 = layer_sizes

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.output = nn.Linear(fc2, num_atoms)
        initialize_weights(self, weight_low, weight_high)

    def forward(self, state, actions, log=False):
        """
        Build a network that estimates the value of an action given the state.
        """

        x = F.relu(self.fc1(state))
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc2(x))
        logits = self.output(x)
        # Only calculate the type of softmax needed by the foward call, to save
        # a modest amount of calculation across 1000s of timesteps.
        if log:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)
