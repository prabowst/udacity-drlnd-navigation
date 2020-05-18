import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, obs_size, action_size, seed):

        '''Initialize the Q-network architecture with 2 hidden layers
           with the input of obs_size and output of action_size

        Params:
            action_size     : size of the action sample
            obs_size        : size of the observation state / states
            n_hidden        : number of neurons in hidden layers
        '''

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, state):

        '''Define the forward pass of the neural network using ReLU activation
           functions and no activation function at the last layer

        Params:
            state           : the current state of the environment

        Return:
            x               : action of the agent
        '''

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
