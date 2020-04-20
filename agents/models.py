import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class FCQNetwork(nn.Module):
    """Fully connected DNN Q function which outputs array of action values"""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units, fc2_units) # action input from second fc layer
        self.fc3  = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CNNQNetwork(nn.Module):
    """CNN Q Function, which outputs array of action values"""

    def __init__(self, state_size, action_size, seed, conv1_filters=16, conv2_filters=16, conv3_filters=16, fc1_units=200, fc2_units=200):
        """Initialize parameters and build model.
        Params
        ======
            state_size (list): Shape of each state image, e.g [3, 28, 28] 
            action_size (int): Dimension of each action
            seed (int): Random seed
            conv1_filters (int): Number of filters for first CNN layer
            conv2_filters (int): Number of filters for second CNN layer
            conv3_filters (int): Number of filters for third CNN layer
            fc1_units (int): Number of nodes in first FC layer
            fc2_units (int): Number of nodes in second FC layer
        """
        super(CNNQNetwork, self).__init__()
        self.seed  = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(state_size[0], conv1_filters, 3, padding=1)       
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, 3, padding=1)

        self.fc1   = nn.Linear(conv3_filters*state_size[1]*state_size[2], fc1_units) # action input from first fc layer
        self.drop  = nn.Dropout(p=0.4)
        self.fc2   = nn.Linear(fc1_units, fc2_units)
        self.fc3   = nn.Linear(fc2_units, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))

        return self.fc3(x)

class FCCritic(nn.Module):
    """Fully connected DNN Critics Q Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1  = nn.Linear(state_size, fc1_units)
        self.fc2  = nn.Linear(fc1_units + action_size, fc2_units) # action input from second fc layer
        self.fc3  = nn.Linear(fc2_units, fc3_units)
        self.fc4  = nn.Linear(fc3_units, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)

        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

class CNNCritic(nn.Module):
    """CNN Critics Q Model. Implement based on DDPG paper 2016"""

    def __init__(self, state_size, action_size, seed, conv1_filters=32, conv2_filters=32, conv3_filters=32, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (list): Shape of each state image, e.g [3, 28, 28] 
            action_size (int): Dimension of each action
            seed (int): Random seed
            conv1_filters (int): Number of filters for first CNN layer
            conv2_filters (int): Number of filters for second CNN layer
            conv3_filters (int): Number of filters for third CNN layer
            fc1_units (int): Number of nodes in first FC layer
            fc2_units (int): Number of nodes in second FC layer
        """
        super(CNNCritic, self).__init__()
        self.seed  = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(state_size[0], conv1_filters, 3, padding=1)       
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, 3, padding=1)

        self.fc1   = nn.Linear(conv3_filters*state_size[1]*state_size[2] + action_size, fc1_units) # action input from first fc layer
        self.drop  = nn.Dropout(p=0.4)
        self.fc2   = nn.Linear(fc1_units, fc2_units)
        self.fc3   = nn.Linear(fc2_units, 1)
        

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        xs = F.relu(self.conv1(state))
        xs = F.relu(self.conv2(xs))
        xs = F.relu(self.conv3(xs))

        xs = xs.view(x.size(0), -1)
        x = torch.cat((xs, action), dim=1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class FCPolicy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCPolicy, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.tanh(self.fc2(x))

class CNNPolicy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv1_filters=32, conv2_filters=32, conv3_filters=32, fc1_units=200, fc2_units=200):
        """Initialize parameters and build model.
        Params
        ======
            state_size (list): Shape of each state image, e.g [3, 28, 28] 
            action_size (int): Dimension of each action
            seed (int): Random seed
            conv1_filters (int): Number of filters for first CNN layer
            conv2_filters (int): Number of filters for second CNN layer
            conv3_filters (int): Number of filters for third CNN layer
            fc1_units (int): Number of nodes in first FC layer
            fc2_units (int): Number of nodes in second FC layer
        """
        super(CNNPolicy, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(state_size[0], conv1_filters, 3, padding=1)       
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, 3, padding=1)

        self.fc1   = nn.Linear(conv3_filters*state_size[1]*state_size[2], fc1_units) # action input from first fc layer
        self.drop  = nn.Dropout(p=0.4)
        self.fc2   = nn.Linear(fc1_units, fc2_units)
        self.fc3   = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))

        return F.softmax(self.fc3(x), dim=1)



class MAFCPolicy(nn.Module) :
    r"""A simple deterministic policy network with batch norms
    Args:
        observationShape (tuple): shape of the observations given to the network
        actionShape (tuple): shape of the actions to be computed by the network
    """
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128) :
        super(MAFCPolicy, self).__init__()

        self.seed     = torch.manual_seed(seed)

        self.bn_input = nn.BatchNorm1d(state_size)
        self.fc1      = nn.Linear(state_size, fc1_units)
        self.bn_fc1   = nn.BatchNorm1d(fc1_units)
        self.fc2      = nn.Linear(fc1_units, fc2_units)
        self.bn_fc2   = nn.BatchNorm1d(fc2_units)
        self.fc3      = nn.Linear(fc2_units, action_size)
        self.reset_parameters()


    def reset_parameters(self) :
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state) :
        r"""Forward pass for this deterministic policy
        Args:
            state (torch.tensor): observation used to decide the action
        """
        x = self.bn_input(state)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))

        return F.tanh(self.fc3(x))

class MAFCCritic(nn.Module) :
    r"""A simple critic Q-network with batch norm to be used for the centralized critics
    Args:
        joint_state_size (tuple): shape of the augmented state representation [o1,o2,...on]
        joint_action_size (tuple): shape of the augmented action representation [a1,a2,...,an]
    """
    def __init__( self, joint_state_size, joint_action_size, seed, fc1_units=128, fc2_units=128) :
        super(MAFCCritic, self).__init__()

        self.seed     = torch.manual_seed(seed)

        self.bn_input = nn.BatchNorm1d(joint_state_size)
        self.fc1      = nn.Linear(joint_state_size, fc1_units)
        self.fc2      = nn.Linear(fc1_units + joint_action_size, fc2_units)
        self.fc3      = nn.Linear(fc2_units, 1)
        self.reset_parameters()


    def reset_parameters(self) :
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, joint_states, joint_actions) :
        r"""Forward pass for this critic at a given (x=[o1,...,an],a=[a1...an]) pair
        Args:
            joint_states (torch.tensor): augmented observation [o1,o2,...,on]
            joint_actions (torch.tensor): augmented action [a1,a2,...,an]
        """
        xs = self.bn_input(joint_states)
        xs = F.relu(self.fc1(xs))

        x = torch.cat([xs, joint_actions], dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)