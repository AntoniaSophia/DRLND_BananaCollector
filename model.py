import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ Actor Model (Policy-based) """
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ Init the neural network
        
            state_size: input dimension (we're inputting states into the NN, so we're calling it state_size)
            action_size: action dimension (output dimension - output layer of the NN)
            seed: Random seed, to keep reproducability across experiment
            fc1_units: Number of nodes in first hidden layer
            fc2_units: Number of nodes in second hidden layer
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units) # hidden layer #1
        self.fc2 = nn.Linear(fc1_units, fc2_units) # hidden layer #2
        self.fc3 = nn.Linear(fc2_units, action_size) # output layer
    
    
    def forward(self, state):
        """ Propagate state through the neural network (arriving at the action-tensors, returning the output layer) """
       
        # state -> fc1
        x = F.relu(self.fc1(state))
        
        # fc1 -> fc2
        x = F.relu(self.fc2(x))
        
        # fc2 -> fc3 (output)
        return self.fc3(x)
