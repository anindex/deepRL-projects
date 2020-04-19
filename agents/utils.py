import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(*x.shape) 
        self.state = x + dx
        return self.state

class NormalNoise():
    """A simple normal-noise sampler
        
    Args:
        size (tuple): size of the noise to be generated
        seed (int): random seed for the rnd-generator
        sigma (float): standard deviation of the normal distribution
    """
    def __init__(self, shape, seed, sigma = 0.2):
        self.shape = shape
        self.seed  = np.random.seed(seed)
        self.sigma = sigma

    def reset(self):
        pass

    def sample(self):
        res = self.sigma * np.random.randn(*self.shape)
        return res

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, discrete_action=True):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if discrete_action:
            actions = actions.long()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MAReplayBuffer:
    """Fixed-size buffer to store multi-agents experience tuples."""

    def __init__(self, buffer_size, batch_size, num_agents, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new multi-agent experience to memory."""
        assert len(states)      == self.num_agents, 'ERROR> group states size mismatch'
        assert len(actions)     == self.num_agents, 'ERROR> group actions size mismatch'
        assert len(rewards)     == self.num_agents, 'ERROR> group rewards size mismatch'
        assert len(next_states) == self.num_agents, 'ERROR> group next states size mismatch'
        assert len(dones)       == self.num_agents, 'ERROR> group dones size mismatch'

        experience = (states, actions, rewards, next_states, dones)
        self.memory.append(experience)
    
    def sample(self, discrete_action=True):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor([e[0] for e in experiences], dtype=torch.float).to(device)
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.float).to(device)
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float).unsqueeze(2).to(device)
        next_states = torch.tensor([e[3] for e in experiences], dtype=torch.float).to(device)
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.float).unsqueeze(2).to(device)

        if discrete_action:
            actions = actions.long()
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)