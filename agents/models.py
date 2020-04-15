import torch
import torch.nn as nn
import torch.nn.functional as F

class FCQNetwork(nn.Module):
    """Fully connected DNN Critics Q Model."""

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
        self.fc2  = nn.Linear(fc1_units + action_size, fc2_units) # action input from second fc layer
        self.fc3  = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)

        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CNNQNetwork(nn.Module):
    """CNN Critics Q Model. Implement based on DDPG paper 2016"""

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
        super(CNNQNetwork, self).__init__()
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

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)

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