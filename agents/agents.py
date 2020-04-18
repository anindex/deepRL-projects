from .models import FCQNetwork, CNNQNetwork, FCPolicy, CNNPolicy, FCCritic, CNNCritic, MAFCPolicy, MAFCCritic
from .utils import OUNoise, NormalNoise, ReplayBuffer, MAReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import numpy as np
import random

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.98            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_EVERY = 20       # how often to update the network
NUM_UPDATES  = 7        # number of updates

MA_BUFFER_SIZE = int(1e6)  # replay buffer size
MA_BATCH_SIZE = 256        # minibatch size
MA_GAMMA = 0.999           # discount factor
MA_TAU = 1e-3              # for soft update of target parameters
MA_LR = 5e-4               # learning rate 
MA_LR_ACTOR = 4e-4         # learning rate of the actor 
MA_LR_CRITIC = 8e-4        # learning rate of the critic
MA_EPSILON_DECAY = 5e-6    # decay factor for e-greedy linear schedule
MA_WEIGHT_DECAY = 0.0      # L2 weight decay
MA_UPDATE_EVERY = 1        # how often to update the network
MA_NUM_UPDATES  = 1        # number of updates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, pixel_input=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int/list): dimension of each state, if it is pixelated input then state_size is image shape
            action_size (int): dimension of each action
            seed (int): random seed
            pixel_input (boolean): flag of low or high dimensional input
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if pixel_input:
            self.qnetwork_local = CNNQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = CNNQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = FCQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = FCQNetwork(state_size, action_size, seed).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents=1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = FCPolicy(state_size, action_size, random_seed).to(device)
        self.actor_target = FCPolicy(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = FCCritic(state_size, action_size, random_seed).to(device)
        self.critic_target = FCCritic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noises = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # counting steps
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _  in range(NUM_UPDATES):
                experiences = self.memory.sample(discrete_action=False)
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noises.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noises.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # clipping gradient to 1 for stable learning
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class MADDPGAgent():
    """Interacts with and learns from the multi-agents environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents=2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_locals = [MAFCPolicy(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_targets = [MAFCPolicy(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor_local.parameters(), lr=MA_LR_ACTOR) for actor_local in self.actor_locals]

        summary(self.actor_locals[0], (num_agents, state_size))
        print(self.actor_locals[0])

        # Critic Network (w/ Target Network)
        self.critic_locals = [MAFCCritic(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critic_targets = [MAFCCritic(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critic_optimizers = [optim.Adam(critic_local.parameters(), lr=MA_LR_CRITIC) for critic_local in self.critic_locals]

        summary(self.critic_locals[0], [(num_agents * state_size, ), (num_agents * action_size, )])
        print(self.critic_locals[0])

        # Noise process
        self.noises = NormalNoise(action_size, random_seed)

        # Replay memory
        self.memory = MAReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, num_agents, random_seed)

        # counting steps
        self.t_step = 0
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        self.t_step = (self.t_step + 1) % MA_UPDATE_EVERY
        # Learn, if enough samples are available in memory
        if len(self.memory) > MA_BATCH_SIZE and self.t_step == 0:
            for _  in range(MA_NUM_UPDATES):
                self.learn(MA_GAMMA)

    def act(self, states, eps=0., add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        for actor_local in self.actor_locals:
            actor_local.eval()

        with torch.no_grad():
            actions = []
            for i, actor_local in enumerate(self.actor_locals):
                i_state = torch.from_numpy(states[i]).float().to(device)
                i_action = actor_local(i_state).cpu().data.numpy().squeeze()
                actions.append(i_action)
            actions = np.array(actions)

            if add_noise:
                actions += np.array([eps * self.noises.sample() for _ in range(self.num_agents)]).reshape(actions.shape)

        for actor_local in self.actor_locals:
            actor_local.train()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noises.reset()

    def learn(self, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            gamma (float): discount factor
        """
        # for each agents, sample a batch memory for learning
        for i in range(self.num_agents):
            states, actions, rewards, next_states, dones = self.memory.sample(discrete_action=False)

            # preprocess dimension for joint states and joint actions for computing Q Critic
            joint_states      = states.reshape(states.shape[0], -1)
            joint_next_states = states.reshape(next_states.shape[0], -1)
            joint_actions     = actions.reshape(actions.shape[0], -1)

            # preprocess dimension for local states, rewards and dones
            local_states      = states[:, i, :]
            local_next_states = next_states[:, i, :]
            local_rewards     = rewards[:, i, :]
            local_dones       = dones[:, i, :]

            # ---------------------------- update critic ---------------------------- #
            # Get predicted joint next-state actions and Q values from target models
            with torch.no_grad():
                joint_next_actions = torch.stack([self.actor_targets[i_actor](local_next_states) for i_actor in range(self.num_agents), dim=1)
                joint_next_actions = joint_next_actions.reshape(joint_next_actions.shape[0], -1)

                # Compute Q targets for current states (y_i)
                Q_targets = local_rewards + gamma * (1 - local_dones) * self.critic_targets[i](joint_next_states, joint_next_actions)
        
            # Compute critic loss
            Q_expected = self.critic_locals[i](states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_locals[i].parameters(), 1) # clipping gradient to 1 for stable learning
            self.critic_optimizers[i].step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = torch.stack([self.actor_locals[i](local_states) if i == i_actor else actions[:, i_actor, :] for i_actor in range(self.num_agents)], dim=1)
            actions_pred = actions_pred.reshape(actions_pred.shape[0], -1)
            actor_loss = -self.critic_locals[i](joint_states, actions_pred).mean()

            # Minimize the loss
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_locals[i], self.critic_targets[i], MA_TAU)
            self.soft_update(self.actor_locals[i], self.actor_targets[i], MA_TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)