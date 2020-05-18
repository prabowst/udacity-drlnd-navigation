import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

from dqnbanana import QNetwork

class DQN:

    def __init__(self, size_state, size_action, lr, tau, DDQN, seed):

        '''Initialize the DQN class along with the DQN networks

        Params:
            size_state (int)    : size of the state or observation
            size_action (int)   : size of the possible actions to select from
            gamma (float)       : discount rate
            lr (float)          : learning rate of the network
            tau (float)         : soft update tau constant
            n_hidden (int)      : number of neurons in the hidden layers
            seed (int)          : random seed number
        '''

        self.size_state = size_state
        self.size_action = size_action
        self.gamma = 0.99
        self.lr = lr
        self.tau = tau
        self.DDQN = DDQN
        self.batch_size = 64
        self.targetUpdateNet = 4
        self.seed = random.seed(seed)

        # GPU enabled if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        # Q network, local and target
        self.qnet_local = QNetwork(self.size_state, self.size_action,
                                   seed).to(self.device)
        self.qnet_target = QNetwork(self.size_state, self.size_action,
                                    seed).to(self.device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=self.lr)

        # Setup tuple for replay buffer
        self.experience_replay = deque(maxlen=100000)
        labels = ['state', 'action', 'reward', 'state_', 'done']
        self.experience = namedtuple('Experience', field_names=labels)
        self.t_step = 0     # counter for update

    def step(self, state, action, reward, state_, done):

        '''Learn for every step fulfilled by targetUpdateNet after appending
           memory experience

        Params:
            state (array_like)  : current state
            action (array_like) : action taken
            reward (array_like) : reward for the specific action taken
            state_ (array_like) : new state after action is executed
            done (array_like)   : status of episode (finished or not)
        '''

        # Append the experience
        exp = self.experience(state, action, reward, state_, done)
        self.experience_replay.append(exp)

        # Increment time-step and look for learning opportunities
        self.t_step = self.t_step + 1
        if self.t_step % self.targetUpdateNet == 0:
            if len(self.experience_replay) < self.batch_size:
                return

            experience_batch = self.sample_replay()
            self.learn(experience_batch)

    def select_action(self, state, eps=0.):

        '''Returns actions based on the current state of the environment using
           the current policy

        Params:
            state (array_like)  : current state
            eps (float)         : epsilon for epsilon-greedy action

        Return:
            action selected
        '''

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            predict_action = self.qnet_local(state)
        self.qnet_local.train()

        if np.random.rand() < eps:
            return random.choice(np.arange(self.size_action))
        else:
            return np.argmax(predict_action.cpu().data.numpy())

    def sample_replay(self):

        '''Take random sample of experience from the batches available within
           the replay buffer

        Return:
            tuple of states, actions, rewards, next states and dones
        '''

        experiences = random.sample(self.experience_replay, self.batch_size)

        def format_torch(input):
            return torch.from_numpy(np.vstack(input))
        def format_uint8(input):
            return torch.from_numpy(np.vstack(input).astype(np.uint8))

        states = format_torch([exp.state for exp in experiences if exp is not None]).float()
        actions = format_torch([exp.action for exp in experiences if exp is not None]).long()
        rewards = format_torch([exp.reward for exp in experiences if exp is not None]).float()
        states_ = format_torch([exp.state_ for exp in experiences if exp is not None]).float()
        dones = format_uint8([exp.done for exp in experiences if exp is not None]).float()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_ = states_.to(self.device)
        dones = dones.to(self.device)

        return (states, actions, rewards, states_, dones)

    def learn(self, experience_batch):

        '''Setup the qnet to learn from qnet_local and use the qnet_target as
           the Q_target to learn from

        Params:
            experiences_batch (array_like)  : a batch of memory replay tuples
        '''

        states, actions, rewards, states_, dones = experience_batch

        if self.DDQN == True:
            Q_argmax = self.qnet_local(states_).detach().max(1)[1].unsqueeze(1)
            Q_targets_ = self.qnet_target(states_).gather(1, Q_argmax)
        else:
            Q_targets_ = self.qnet_target(states_).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (self.gamma*Q_targets_*(1 - dones))
        Q_expect = self.qnet_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expect, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnet_local, self.qnet_target)

    def soft_update(self, local, target):

        '''Carry out the soft update of the network using the constant tau

        Params:
            local (PyTorch model)   : qnet_local model
            target (PyTorch model)  : qnet_target model
        '''

        for local_param, target_param in zip(local.parameters(),
                                             target.parameters()):
            target_param.data.copy_(self.tau * local_param.data + \
                                    (1.0-self.tau) * target_param.data)
